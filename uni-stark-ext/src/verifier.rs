use alloc::vec;
use alloc::vec::Vec;
use core::mem;

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
    PcsError, Proof, StarkGenericConfig, Val, VerifierConstraintFolder, eval_log_up,
    get_symbolic_constraints, interaction_chunks, max_degree,
};

#[derive(Clone, Debug)]
pub struct VerifierInput<Val, A> {
    pub(crate) air: A,
    pub(crate) public_values: Vec<Val>,
    pub(crate) log_quotient_degree: usize,
    pub(crate) constraint_count: usize,
    pub(crate) interaction_count: usize,
    pub(crate) interaction_chunks: Vec<Vec<usize>>,
    pub(crate) max_bus_index: usize,
    pub(crate) max_field_count: usize,
}

impl<Val: Field, A> VerifierInput<Val, A> {
    pub fn new(air: A, public_values: Vec<Val>) -> Self
    where
        A: Air<SymbolicAirBuilder<Val>>,
    {
        let (constraints, interactions) =
            get_symbolic_constraints::<Val, A>(&air, 0, public_values.len());
        let constraint_degree = max_degree(&constraints);
        let log_quotient_degree = log2_ceil_usize(constraint_degree.saturating_sub(1));
        let interaction_chunks = interaction_chunks((1 << log_quotient_degree) + 1, &interactions);
        let constraint_count = constraints.len() + interaction_chunks.len() + 3;
        let interaction_count = interactions.len();
        let max_bus_index =
            itertools::max(interactions.iter().map(|i| i.bus_index)).unwrap_or_default();
        let max_field_count =
            itertools::max(interactions.iter().map(|i| i.fields.len())).unwrap_or_default();
        Self {
            air,
            public_values,
            log_quotient_degree,
            constraint_count,
            interaction_count,
            interaction_chunks,
            max_bus_index,
            max_field_count,
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
        log_up_sums,
        opened_values,
        opening_proof,
        log_degrees,
    } = proof;

    let valid_shape = opened_values.len() == inputs.len()
        && izip!(&inputs, opened_values).all(|(input, opened_values)| {
            opened_values.main_local.len() == input.air.width()
                && opened_values.main_next.len() == input.air.width()
                && opened_values.log_up_local.len()
                    == <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION
                        * (input.interaction_chunks.len() + 1)
                && opened_values.log_up_next.len()
                    == <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION
                        * (input.interaction_chunks.len() + 1)
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

    let beta: SC::Challenge = challenger.sample_algebra_element();
    let gamma: SC::Challenge = challenger.sample_algebra_element();

    if log_up_sums.iter().copied().sum::<SC::Challenge>() != SC::Challenge::ZERO {
        return Err(VerificationError::InvalidLogUpSum);
    }

    log_up_sums
        .iter()
        .for_each(|sum| challenger.observe_algebra_element(*sum));

    challenger.observe(commitments.log_up_chunks.clone());

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
                commitments.log_up_chunks.clone(),
                izip!(&main_domains, opened_values)
                    .map(|(main_domain, opened_values)| {
                        let zeta_next = main_domain.next_point(zeta).unwrap();
                        (
                            *main_domain,
                            vec![
                                (zeta, opened_values.log_up_local.clone()),
                                (zeta_next, opened_values.log_up_next.clone()),
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

    let max_bus_index =
        itertools::max(inputs.iter().map(|input| input.max_bus_index)).unwrap_or_default();
    let max_field_count =
        itertools::max(inputs.iter().map(|input| input.max_field_count)).unwrap_or_default();
    let beta_powers = beta.powers().skip(1).take(max_field_count).collect_vec();
    let gamma_powers = gamma.powers().skip(1).take(max_bus_index + 1).collect_vec();

    izip!(
        &inputs,
        main_domains,
        quotient_chunks_domains,
        log_up_sums,
        opened_values,
    )
    .try_for_each(
        |(input, main_domain, quotient_chunks_domains, log_up_sum, opened_values)| {
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

            let [log_up_local, log_up_next] =
                [&opened_values.log_up_local, &opened_values.log_up_next].map(|values| {
                    values
                        .chunks(SC::Challenge::DIMENSION)
                        .map(|ch| {
                            ch.iter()
                                .enumerate()
                                .map(|(e_i, &c)| SC::Challenge::ith_basis_element(e_i) * c)
                                .sum::<SC::Challenge>()
                        })
                        .collect_vec()
                });

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
                beta_powers: &beta_powers,
                gamma_powers: &gamma_powers,
                numers: vec![Default::default(); input.interaction_count],
                denoms: vec![Default::default(); input.interaction_count],
                interaction_index: 0,
            };
            input.air.eval(&mut folder);

            let numers = mem::take(&mut folder.numers);
            let denoms = mem::take(&mut folder.denoms);
            eval_log_up(
                &mut folder,
                &numers,
                &denoms,
                &log_up_local,
                &log_up_next,
                &input.interaction_chunks,
                *log_up_sum,
            );

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
    InvalidLogUpSum,
    /// An error occurred while verifying the claimed openings.
    InvalidOpeningArgument(PcsErr),
    /// Out-of-domain evaluation mismatch, i.e. `constraints(zeta)` did not match
    /// `quotient(zeta) Z_H(zeta)`.
    OodEvaluationMismatch,
}
