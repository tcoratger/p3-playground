use alloc::vec;
use alloc::vec::Vec;
use core::ops::Deref;

use itertools::{Itertools, izip};
use p3_air::Air;
use p3_challenger::{CanObserve, CanSample, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{BasedVectorSpace, Field, PackedValue, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::{debug_span, info_span, instrument};

use crate::{
    Commitments, Domain, OpenedValues, PackedChallenge, PackedVal, Proof, ProverConstraintFolder,
    StarkGenericConfig, SymbolicAirBuilder, Val, VerifierInput,
};

#[derive(Clone, Debug)]
pub struct ProverInput<Val, A> {
    inner: VerifierInput<Val, A>,
    main_trace: RowMajorMatrix<Val>,
}

impl<Val, A> Deref for ProverInput<Val, A> {
    type Target = VerifierInput<Val, A>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<Val: Field, A> ProverInput<Val, A> {
    pub fn new(air: A, public_values: Vec<Val>, main_trace: RowMajorMatrix<Val>) -> Self
    where
        A: Air<SymbolicAirBuilder<Val>>,
    {
        Self {
            inner: VerifierInput::new(air, public_values),
            main_trace,
        }
    }
}

#[instrument(skip_all)]
#[allow(clippy::multiple_bound_locations)] // cfg not supported in where clauses?
pub fn prove<
    SC,
    #[cfg(debug_assertions)] A: for<'a> Air<crate::check_constraints::DebugConstraintBuilder<'a, Val<SC>>>,
    #[cfg(not(debug_assertions))] A,
>(
    config: &SC,
    inputs: Vec<ProverInput<Val<SC>, A>>,
    challenger: &mut SC::Challenger,
) -> Proof<SC>
where
    SC: StarkGenericConfig,
    A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<ProverConstraintFolder<'a, SC>>,
{
    #[cfg(debug_assertions)]
    inputs.iter().for_each(|input| {
        crate::check_constraints::check_constraints(
            &input.inner.air,
            &input.main_trace,
            &input.inner.public_values,
        )
    });

    let (inputs, main_traces, log_degrees) = inputs
        .into_iter()
        .map(|input| {
            let log_degree = log2_strict_usize(input.main_trace.height());
            (input.inner, input.main_trace, log_degree)
        })
        .collect::<(Vec<_>, Vec<_>, Vec<_>)>();

    let pcs = config.pcs();
    let (main_domains, quotient_domains) = izip!(&inputs, &log_degrees)
        .map(|(input, log_degree)| {
            let main_domain = pcs.natural_domain_for_degree(1 << log_degree);
            let quotient_domain =
                main_domain.create_disjoint_domain(1 << (log_degree + input.log_quotient_degree));
            (main_domain, quotient_domain)
        })
        .collect::<(Vec<_>, Vec<_>)>();

    let (main_commit, main_data) = info_span!("commit to main data")
        .in_scope(|| pcs.commit(izip!(main_domains.iter().copied(), main_traces).collect()));

    // Observe the instance.
    // degree < 2^255 so we can safely cast log_degree to a u8.
    log_degrees
        .iter()
        .for_each(|log_degree| challenger.observe(Val::<SC>::from_u8(*log_degree as u8)));
    // TODO: Might be best practice to include other instance data here; see verifier comment.

    challenger.observe(main_commit.clone());
    inputs
        .iter()
        .for_each(|input| challenger.observe_slice(&input.public_values));
    let alpha: SC::Challenge = challenger.sample_algebra_element();

    let max_constraint_count =
        itertools::max(inputs.iter().map(|input| input.constraint_count)).unwrap_or_default();
    let mut alpha_powers = alpha.powers().take(max_constraint_count).collect_vec();
    alpha_powers.reverse();

    let quotient_values = izip!(&inputs, &main_domains, &quotient_domains)
        .enumerate()
        .map(|(idx, (input, main_domain, quotient_domain))| {
            let main_trace_on_quotient_domain =
                pcs.get_evaluations_on_domain(&main_data, idx, *quotient_domain);
            quotient_values(
                &input.air,
                &input.public_values,
                *main_domain,
                *quotient_domain,
                main_trace_on_quotient_domain,
                &alpha_powers[max_constraint_count - input.constraint_count..],
            )
        })
        .collect::<Vec<_>>();

    let quotient_chunks = izip!(&inputs, &quotient_domains, quotient_values)
        .flat_map(|(input, quotient_domain, quotient_values)| {
            let quotient_flat = RowMajorMatrix::new_col(quotient_values).flatten_to_base();
            let quotient_chunks =
                quotient_domain.split_evals(input.quotient_degree(), quotient_flat);
            let qc_domains = quotient_domain.split_domains(input.quotient_degree());
            izip!(qc_domains, quotient_chunks)
        })
        .collect();

    let (quotient_commit, quotient_data) =
        info_span!("commit to quotient poly chunks").in_scope(|| pcs.commit(quotient_chunks));
    challenger.observe(quotient_commit.clone());

    let commitments = Commitments {
        main: main_commit,
        quotient_chunks: quotient_commit,
    };

    let zeta: SC::Challenge = challenger.sample();

    let (opened_values, opening_proof) = info_span!("open").in_scope(|| {
        pcs.open(
            vec![
                (
                    &main_data,
                    log_degrees
                        .iter()
                        .map(|log_degree| {
                            let main_domain = pcs.natural_domain_for_degree(1 << log_degree);
                            let zeta_next = main_domain.next_point(zeta).unwrap();
                            vec![zeta, zeta_next]
                        })
                        .collect(),
                ),
                (
                    &quotient_data,
                    // open every chunk at zeta
                    vec![vec![zeta]; inputs.iter().map(VerifierInput::quotient_degree).sum()],
                ),
            ],
            challenger,
        )
    });
    let mut quotient_chunks = opened_values[1].iter().map(|v| v[0].clone());
    let opened_values = izip!(&inputs, &opened_values[0])
        .map(|(input, main)| OpenedValues {
            main_local: main[0].clone(),
            main_next: main[1].clone(),
            quotient_chunks: quotient_chunks
                .by_ref()
                .take(input.quotient_degree())
                .collect_vec(),
        })
        .collect_vec();
    Proof {
        commitments,
        opened_values,
        opening_proof,
        log_degrees,
    }
}

#[instrument(name = "compute quotient polynomial", skip_all)]
fn quotient_values<SC, A, Mat>(
    air: &A,
    public_values: &Vec<Val<SC>>,
    main_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    main_trace_on_quotient_domain: Mat,
    alpha_powers: &[SC::Challenge],
) -> Vec<SC::Challenge>
where
    SC: StarkGenericConfig,
    A: for<'a> Air<ProverConstraintFolder<'a, SC>>,
    Mat: Matrix<Val<SC>> + Sync,
{
    let quotient_size = quotient_domain.size();
    let width = main_trace_on_quotient_domain.width();
    let mut sels = debug_span!("Compute Selectors")
        .in_scope(|| main_domain.selectors_on_coset(quotient_domain));

    let qdb = log2_strict_usize(quotient_domain.size()) - log2_strict_usize(main_domain.size());
    let next_step = 1 << qdb;

    // We take PackedVal::<SC>::WIDTH worth of values at a time from a quotient_size slice, so we need to
    // pad with default values in the case where quotient_size is smaller than PackedVal::<SC>::WIDTH.
    for _ in quotient_size..PackedVal::<SC>::WIDTH {
        sels.is_first_row.push(Val::<SC>::default());
        sels.is_last_row.push(Val::<SC>::default());
        sels.is_transition.push(Val::<SC>::default());
        sels.inv_vanishing.push(Val::<SC>::default());
    }

    (0..quotient_size)
        .into_par_iter()
        .step_by(PackedVal::<SC>::WIDTH)
        .flat_map_iter(|i_start| {
            let i_range = i_start..i_start + PackedVal::<SC>::WIDTH;

            let is_first_row = *PackedVal::<SC>::from_slice(&sels.is_first_row[i_range.clone()]);
            let is_last_row = *PackedVal::<SC>::from_slice(&sels.is_last_row[i_range.clone()]);
            let is_transition = *PackedVal::<SC>::from_slice(&sels.is_transition[i_range.clone()]);
            let inv_vanishing = *PackedVal::<SC>::from_slice(&sels.inv_vanishing[i_range]);

            let main = RowMajorMatrix::new(
                main_trace_on_quotient_domain.vertically_packed_row_pair(i_start, next_step),
                width,
            );

            let accumulator = PackedChallenge::<SC>::ZERO;
            let mut folder = ProverConstraintFolder {
                main: main.as_view(),
                public_values,
                is_first_row,
                is_last_row,
                is_transition,
                alpha_powers,
                accumulator,
                constraint_index: 0,
            };
            air.eval(&mut folder);

            // quotient(x) = constraints(x) / Z_H(x)
            let quotient = folder.accumulator * inv_vanishing;

            // "Transpose" D packed base coefficients into WIDTH scalar extension coefficients.
            (0..core::cmp::min(quotient_size, PackedVal::<SC>::WIDTH)).map(move |idx_in_packing| {
                SC::Challenge::from_basis_coefficients_fn(|coeff_idx| {
                    quotient.as_basis_coefficients_slice()[coeff_idx].as_slice()[idx_in_packing]
                })
            })
        })
        .collect()
}
