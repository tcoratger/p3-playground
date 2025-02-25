use alloc::vec;
use alloc::vec::Vec;

use itertools::{Itertools, izip};
use p3_air::Air;
use p3_challenger::{CanObserve, CanSample, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;
use p3_util::log2_ceil_usize;
use tracing::instrument;

use crate::symbolic_builder::SymbolicAirBuilder;
use crate::{
    PcsError, Proof, StarkGenericConfig, SymbolicExpression, Val, VerifierConstraintFolder,
    get_symbolic_constraints,
};

#[derive(Clone, Debug)]
pub struct VerifierInput<Val, A> {
    pub(crate) air: A,
    pub(crate) public_values: Vec<Val>,
    pub(crate) log_quotient_degree: usize,
    pub(crate) constraint_count: usize,
}

impl<Val: Field, A> VerifierInput<Val, A> {
    pub fn new(air: A, public_values: Vec<Val>) -> Self
    where
        A: Air<SymbolicAirBuilder<Val>>,
    {
        let constraints = get_symbolic_constraints::<Val, A>(&air, 0, public_values.len());
        let constraint_degree =
            itertools::max(constraints.iter().map(SymbolicExpression::degree_multiple))
                .unwrap_or_default();
        let log_quotient_degree = log2_ceil_usize(constraint_degree.saturating_sub(1));
        let constraint_count = constraints.len();
        Self {
            air,
            public_values,
            log_quotient_degree,
            constraint_count,
        }
    }

    pub fn quotient_degree(&self) -> usize {
        1 << self.log_quotient_degree
    }
}

#[instrument(skip_all)]
pub fn verify<SC, A>(
    config: &SC,
    inputs: Vec<VerifierInput<Val<SC>, A>>,
    challenger: &mut SC::Challenger,
    proof: &Proof<SC>,
) -> Result<(), VerificationError<PcsError<SC>>>
where
    SC: StarkGenericConfig,
    A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<VerifierConstraintFolder<'a, SC>>,
{
    let Proof {
        commitments,
        opened_values,
        opening_proof,
        log_degrees,
    } = proof;

    let valid_shape = opened_values.len() == inputs.len()
        && izip!(&inputs, opened_values).all(|(input, opened_values)| {
            opened_values.main_local.len() == input.air.width()
                && opened_values.main_next.len() == input.air.width()
                && opened_values.quotient_chunks.len() == input.quotient_degree()
                && opened_values
                    .quotient_chunks
                    .iter()
                    .all(|qc| qc.len() == <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION)
        })
        && log_degrees.len() == inputs.len();
    if !valid_shape {
        return Err(VerificationError::InvalidProofShape);
    }

    let pcs = config.pcs();
    let (main_domains, quotient_chunks_domains) = izip!(&inputs, log_degrees)
        .map(|(input, log_degree)| {
            let main_domain = pcs.natural_domain_for_degree(1 << log_degree);
            let quotient_domain =
                main_domain.create_disjoint_domain(1 << (log_degree + input.log_quotient_degree));
            let quotient_chunks_domains = quotient_domain.split_domains(input.quotient_degree());
            (main_domain, quotient_chunks_domains)
        })
        .collect::<(Vec<_>, Vec<_>)>();

    // Observe the instance.
    log_degrees
        .iter()
        .for_each(|log_degree| challenger.observe(Val::<SC>::from_u8(*log_degree as u8)));
    // TODO: Might be best practice to include other instance data here in the transcript, like some
    // encoding of the AIR. This protects against transcript collisions between distinct instances.
    // Practically speaking though, the only related known attack is from failing to include public
    // values. It's not clear if failing to include other instance data could enable a transcript
    // collision, since most such changes would completely change the set of satisfying witnesses.

    challenger.observe(commitments.main.clone());
    inputs
        .iter()
        .for_each(|input| challenger.observe_slice(&input.public_values));
    let alpha: SC::Challenge = challenger.sample_algebra_element();

    challenger.observe(commitments.quotient_chunks.clone());

    let zeta: SC::Challenge = challenger.sample();

    pcs.verify(
        vec![
            (
                commitments.main.clone(),
                izip!(&main_domains, opened_values)
                    .map(|(main_domain, opened_values)| {
                        let zeta_next = main_domain.next_point(zeta).unwrap();
                        (
                            *main_domain,
                            vec![
                                (zeta, opened_values.main_local.clone()),
                                (zeta_next, opened_values.main_next.clone()),
                            ],
                        )
                    })
                    .collect(),
            ),
            (
                commitments.quotient_chunks.clone(),
                izip!(&quotient_chunks_domains, opened_values)
                    .flat_map(|(quotient_chunks_domains, opened_values)| {
                        izip!(quotient_chunks_domains, &opened_values.quotient_chunks)
                            .map(|(domain, values)| (*domain, vec![(zeta, values.clone())]))
                    })
                    .collect(),
            ),
        ],
        opening_proof,
        challenger,
    )
    .map_err(VerificationError::InvalidOpeningArgument)?;

    izip!(
        &inputs,
        main_domains,
        quotient_chunks_domains,
        opened_values
    )
    .try_for_each(
        |(input, main_domain, quotient_chunks_domains, opened_values)| {
            let zps = quotient_chunks_domains
                .iter()
                .enumerate()
                .map(|(i, domain)| {
                    quotient_chunks_domains
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, other_domain)| {
                            other_domain.vanishing_poly_at_point(zeta)
                                * other_domain
                                    .vanishing_poly_at_point(domain.first_point())
                                    .inverse()
                        })
                        .product::<SC::Challenge>()
                })
                .collect_vec();

            let quotient = opened_values
                .quotient_chunks
                .iter()
                .enumerate()
                .map(|(ch_i, ch)| {
                    zps[ch_i]
                        * ch.iter()
                            .enumerate()
                            .map(|(e_i, &c)| SC::Challenge::ith_basis_element(e_i) * c)
                            .sum::<SC::Challenge>()
                })
                .sum::<SC::Challenge>();

            let sels = main_domain.selectors_at_point(zeta);

            let main = VerticalPair::new(
                RowMajorMatrixView::new_row(&opened_values.main_local),
                RowMajorMatrixView::new_row(&opened_values.main_next),
            );

            let mut folder = VerifierConstraintFolder {
                main,
                public_values: &input.public_values,
                is_first_row: sels.is_first_row,
                is_last_row: sels.is_last_row,
                is_transition: sels.is_transition,
                alpha,
                accumulator: SC::Challenge::ZERO,
            };
            input.air.eval(&mut folder);
            let folded_constraints = folder.accumulator;

            // Finally, check that
            //     folded_constraints(zeta) / Z_H(zeta) = quotient(zeta)
            if folded_constraints * sels.inv_vanishing != quotient {
                return Err(VerificationError::OodEvaluationMismatch);
            }

            Ok(())
        },
    )
}

#[derive(Debug)]
pub enum VerificationError<PcsErr> {
    InvalidProofShape,
    /// An error occurred while verifying the claimed openings.
    InvalidOpeningArgument(PcsErr),
    /// Out-of-domain evaluation mismatch, i.e. `constraints(zeta)` did not match
    /// `quotient(zeta) Z_H(zeta)`.
    OodEvaluationMismatch,
}
