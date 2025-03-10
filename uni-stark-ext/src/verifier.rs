use alloc::vec;
use alloc::vec::Vec;
use core::mem;

use itertools::{Itertools, chain, izip};
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
    Domain, PcsError, Proof, ProofPerAir, StarkGenericConfig, Val, VerifierConstraintFolder,
    eval_log_up, get_symbolic_constraints, interaction_chunks, max_degree,
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

impl<Val: Field, A: Air<SymbolicAirBuilder<Val>>> VerifierInput<Val, A> {
    pub fn new(air: A, public_values: Vec<Val>) -> Self {
        let (constraints, interactions) =
            get_symbolic_constraints::<Val, A>(&air, 0, public_values.len());
        let constraint_degree = max_degree(&constraints);
        let log_quotient_degree = log2_ceil_usize(constraint_degree.saturating_sub(1));
        let interaction_chunks = interaction_chunks((1 << log_quotient_degree) + 1, &interactions);
        let log_up_constraint_count = if interactions.is_empty() {
            0
        } else {
            interaction_chunks.len() + 3
        };
        let constraint_count = constraints.len() + log_up_constraint_count;
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

    pub fn has_interaction(&self) -> bool {
        self.interaction_count != 0
    }

    pub fn valid_shape_per_air<SC: StarkGenericConfig>(&self, per_air: &ProofPerAir<SC>) -> bool
    where
        Domain<SC>: PolynomialSpace<Val = Val>,
    {
        let ProofPerAir {
            log_up_sum,
            opened_values,
            ..
        } = per_air;
        let dimension = <SC::Challenge as BasedVectorSpace<Val>>::DIMENSION;
        let log_up_width = self
            .has_interaction()
            .then(|| self.interaction_chunks.len() + 1)
            .unwrap_or_default();
        self.has_interaction() == log_up_sum.is_some()
            && opened_values.main_local.len() == self.air.width()
            && opened_values.main_next.len() == self.air.width()
            && opened_values.log_up_local.len() == dimension * log_up_width
            && opened_values.log_up_next.len() == dimension * log_up_width
            && opened_values.quotient_chunks.len() == self.quotient_degree()
            && opened_values
                .quotient_chunks
                .iter()
                .all(|qc| qc.len() == dimension)
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
        per_air,
        opening_proof,
    } = proof;

    let has_any_interaction = inputs.iter().any(VerifierInput::has_interaction);

    let valid_shape = per_air.len() == inputs.len()
        && izip!(&inputs, per_air).all(|(input, per_air)| input.valid_shape_per_air(per_air))
        && (has_any_interaction == commitments.log_up_chunks.is_some());
    if !valid_shape {
        return Err(VerificationError::InvalidProofShape);
    }

    let pcs = config.pcs();
    let (main_domains, quotient_chunks_domains) = izip!(&inputs, per_air)
        .map(|(input, per_air)| {
            let main_domain = pcs.natural_domain_for_degree(1 << per_air.log_degree);
            let quotient_domain = main_domain
                .create_disjoint_domain(1 << (per_air.log_degree + input.log_quotient_degree));
            let quotient_chunks_domains = quotient_domain.split_domains(input.quotient_degree());
            (main_domain, quotient_chunks_domains)
        })
        .collect::<(Vec<_>, Vec<_>)>();

    // Observe the instance.
    per_air
        .iter()
        .for_each(|per_air| challenger.observe(Val::<SC>::from_u8(per_air.log_degree as u8)));
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

    if per_air
        .iter()
        .flat_map(|per_air| per_air.log_up_sum)
        .sum::<SC::Challenge>()
        != SC::Challenge::ZERO
    {
        return Err(VerificationError::InvalidLogUpSum);
    }

    per_air
        .iter()
        .flat_map(|per_air| per_air.log_up_sum)
        .for_each(|sum| challenger.observe_algebra_element(sum));

    if let Some(log_up_chunks) = &commitments.log_up_chunks {
        challenger.observe(log_up_chunks.clone());
    }

    let alpha: SC::Challenge = challenger.sample_algebra_element();

    challenger.observe(commitments.quotient_chunks.clone());

    let zeta: SC::Challenge = challenger.sample();

    pcs.verify(
        chain![
            Some((
                commitments.main.clone(),
                izip!(&main_domains, per_air)
                    .map(|(main_domain, per_air)| {
                        let zeta_next = main_domain.next_point(zeta).unwrap();
                        (
                            *main_domain,
                            vec![
                                (zeta, per_air.opened_values.main_local.clone()),
                                (zeta_next, per_air.opened_values.main_next.clone()),
                            ],
                        )
                    })
                    .collect(),
            )),
            commitments.log_up_chunks.clone().map(|log_up_chunks| {
                (
                    log_up_chunks,
                    izip!(&inputs, &main_domains, per_air)
                        .filter(|(input, _, _)| input.has_interaction())
                        .map(|(_, main_domain, per_air)| {
                            let zeta_next = main_domain.next_point(zeta).unwrap();
                            (
                                *main_domain,
                                vec![
                                    (zeta, per_air.opened_values.log_up_local.clone()),
                                    (zeta_next, per_air.opened_values.log_up_next.clone()),
                                ],
                            )
                        })
                        .collect(),
                )
            }),
            Some((
                commitments.quotient_chunks.clone(),
                izip!(&quotient_chunks_domains, per_air)
                    .flat_map(|(quotient_chunks_domains, per_air)| {
                        izip!(
                            quotient_chunks_domains,
                            &per_air.opened_values.quotient_chunks
                        )
                        .map(|(domain, values)| (*domain, vec![(zeta, values.clone())]))
                    })
                    .collect(),
            )),
        ]
        .collect(),
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

    izip!(&inputs, main_domains, quotient_chunks_domains, per_air).try_for_each(
        |(input, main_domain, quotient_chunks_domains, per_air)| {
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

            let quotient = per_air
                .opened_values
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
                RowMajorMatrixView::new_row(&per_air.opened_values.main_local),
                RowMajorMatrixView::new_row(&per_air.opened_values.main_next),
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

            if let Some(log_up_sum) = per_air.log_up_sum {
                let numers = mem::take(&mut folder.numers);
                let denoms = mem::take(&mut folder.denoms);
                let [log_up_local, log_up_next] = [
                    &per_air.opened_values.log_up_local,
                    &per_air.opened_values.log_up_next,
                ]
                .map(|values| {
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
                eval_log_up(
                    &mut folder,
                    &numers,
                    &denoms,
                    &log_up_local,
                    &log_up_next,
                    &input.interaction_chunks,
                    log_up_sum,
                );
            }

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
