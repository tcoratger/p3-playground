use alloc::vec::Vec;

use itertools::{Itertools, izip};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{BatchOpeningRef, Mmcs};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix};
use p3_util::reverse_bits_len;

use crate::{CommitPhaseProofStep, FriConfig, FriGenericConfig, FriProof};

#[derive(Debug)]
pub enum FriError<CommitMmcsErr, InputError> {
    InvalidProofShape,
    CommitPhaseMmcsError(CommitMmcsErr),
    InputError(InputError),
    FinalPolyMismatch,
    InvalidPowWitness,
    OpenedRowMismatch,
}

pub fn verify<G, Val, Challenge, M, Challenger>(
    g: &G,
    config: &FriConfig<M>,
    proof: &FriProof<Challenge, M, Challenger::Witness, G::InputProof>,
    challenger: &mut Challenger,
    open_input: impl Fn(
        usize,
        &G::InputProof,
    ) -> Result<Vec<(usize, Challenge)>, FriError<M::Error, G::InputError>>,
) -> Result<(), FriError<M::Error, G::InputError>>
where
    Val: Field,
    Challenge: ExtensionField<Val> + TwoAdicField,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    G: FriGenericConfig<Val, Challenge>,
{
    let betas: Vec<Challenge> = proof
        .commit_phase_commits
        .iter()
        .map(|comm| {
            challenger.observe(comm.clone());
            challenger.sample_algebra_element()
        })
        .collect();

    // Observe all coefficients of the final polynomial.
    proof
        .final_poly
        .iter()
        .for_each(|x| challenger.observe_algebra_element(*x));

    if proof.query_proofs.len() != config.num_queries {
        return Err(FriError::InvalidProofShape);
    }

    // Check PoW.
    if !challenger.check_witness(config.proof_of_work_bits, proof.pow_witness) {
        return Err(FriError::InvalidPowWitness);
    }

    let log_max_height = proof.log_max_height;

    for qp in &proof.query_proofs {
        let index = challenger.sample_bits(log_max_height + g.extra_query_index_bits());
        let ro = open_input(index, &qp.input_proof)?;

        debug_assert!(
            ro.iter().tuple_windows().all(|((l, _), (r, _))| l > r),
            "reduced openings sorted by height descending"
        );

        let folded_eval = verify_query(
            g,
            config,
            index >> g.extra_query_index_bits(),
            izip!(
                &betas,
                &proof.commit_phase_commits,
                &qp.commit_phase_openings
            ),
            ro,
            log_max_height,
        )?;

        let final_poly_index = index >> (proof.commit_phase_commits.len() * config.arity_bits);

        let mut eval = Challenge::ZERO;

        // We open the final polynomial at index `final_poly_index`, which corresponds to evaluating
        // the polynomial at x^k, where x is the 2-adic generator of order `max_height` and k is
        // `reverse_bits_len(final_poly_index, log_max_height)`.
        let x = Challenge::two_adic_generator(log_max_height)
            .exp_u64(reverse_bits_len(final_poly_index, log_max_height) as u64);
        let mut x_pow = Challenge::ONE;

        // Evaluate the final polynomial at x.
        for coeff in &proof.final_poly {
            eval += *coeff * x_pow;
            x_pow *= x;
        }

        if eval != folded_eval {
            return Err(FriError::FinalPolyMismatch);
        }
    }

    Ok(())
}

type CommitStep<'a, F, M> = (
    &'a F,
    &'a <M as Mmcs<F>>::Commitment,
    &'a CommitPhaseProofStep<F, M>,
);

fn verify_query<'a, G, F, EF, M>(
    g: &G,
    config: &FriConfig<M>,
    mut index: usize,
    mut steps: impl Iterator<Item = CommitStep<'a, EF, M>>,
    reduced_openings: Vec<(usize, EF)>,
    log_max_height: usize,
) -> Result<EF, FriError<M::Error, G::InputError>>
where
    F: Field,
    EF: ExtensionField<F>,
    M: Mmcs<EF> + 'a,
    G: FriGenericConfig<F, EF>,
{
    let mut folded_eval = EF::ZERO;
    let mut ro_iter = reduced_openings.into_iter().peekable();

    let mut log_folded_height = log_max_height;

    while log_folded_height > config.log_blowup + config.log_final_poly_len {
        let cur_arity_bits = config.arity_bits.min(log_folded_height);

        if let Some((_, ro)) = ro_iter.next_if(|(lh, _)| *lh == log_folded_height) {
            folded_eval += ro;
        }

        let (beta, comm, opening) = steps.next().unwrap();

        let index_row = index >> cur_arity_bits;

        // Verify that `folded_eval` and evals from reduced_openings match opened_rows
        if opening.opened_rows[0].len() != 1 << cur_arity_bits
            || folded_eval != opening.opened_rows[0][index % (1 << cur_arity_bits)]
        {
            return Err(FriError::OpenedRowMismatch);
        }
        for row in opening.opened_rows.iter().skip(1) {
            let (lh, ro) = ro_iter.next().unwrap();
            // Make sure the read row is of the correct length
            if row.len() != 1 << (lh + cur_arity_bits - log_folded_height) {
                return Err(FriError::OpenedRowMismatch);
            }

            let current_index = index >> (log_folded_height - lh);
            if ro != row[current_index % row.len()] {
                return Err(FriError::OpenedRowMismatch);
            }
        }

        let dims: Vec<_> = opening
            .opened_rows
            .iter()
            .map(|opened_row| Dimensions {
                width: opened_row.len(),
                height: 1 << (log_folded_height - cur_arity_bits),
            })
            .collect();

        config
            .mmcs
            .verify_batch(
                comm,
                &dims,
                index_row,
                BatchOpeningRef::new(&opening.opened_rows, &opening.opening_proof),
            )
            .map_err(FriError::CommitPhaseMmcsError)?;

        // Do the folding logic

        let mut beta = *beta;
        let mut opened_rows_iter = opening.opened_rows.iter().peekable();
        let mut folded_row = opened_rows_iter.next().unwrap().clone();
        let mut index_folded_row = index_row << cur_arity_bits;
        while folded_row.len() > 1 {
            index >>= 1;
            log_folded_height -= 1;
            index_folded_row >>= 1;

            folded_row = fold_partial_row(g, index_folded_row, log_folded_height, beta, folded_row);
            beta = beta.square();

            if let Some(poly_eval) = opened_rows_iter.next_if(|v| v.len() == folded_row.len()) {
                izip!(&mut folded_row, poly_eval).for_each(|(f, v)| *f += *v);
            }
        }

        folded_eval = folded_row.pop().unwrap();

        if !folded_row.is_empty() {
            return Err(FriError::InvalidProofShape);
        }
    }

    if ro_iter.next().is_some() {
        return Err(FriError::InvalidProofShape);
    }
    if steps.next().is_some() {
        return Err(FriError::InvalidProofShape);
    }

    Ok(folded_eval)
}

fn fold_partial_row<G, F, EF>(
    g: &G,
    index: usize,
    log_height: usize,
    beta: EF,
    evals: Vec<EF>,
) -> Vec<EF>
where
    G: FriGenericConfig<F, EF>,
    F: Field,
    EF: ExtensionField<F>,
{
    let folded_matrix = RowMajorMatrix::new(evals, 2);
    folded_matrix
        .rows()
        .enumerate()
        .map(|(i, row)| g.fold_row(index + i, log_height, beta, row.into_iter()))
        .collect()
}
