use alloc::sync::Arc;
use alloc::vec;

use p3_commit::Pcs;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use serde::{Deserialize, Serialize};

use crate::{Com, DynamicAir, PcsProverData, StarkGenericConfig, Val};

/// Holds verifier-specific data for a preprocessed AIR trace.
#[derive(Clone, Serialize, Deserialize)]
pub struct VerifierPrepData<Com> {
    /// Commitment to the preprocessed trace.
    pub commit: Com,
}

impl<Com> From<Com> for VerifierPrepData<Com> {
    /// Converts a commitment into verifier preprocessed data.
    fn from(commit: Com) -> Self {
        Self { commit }
    }
}

/// Holds prover-specific data for a preprocessed AIR trace.
pub struct ProverPrepData<SC: StarkGenericConfig> {
    /// The preprocessed trace matrix.
    pub trace: Arc<RowMajorMatrix<Val<SC>>>,
    /// Prover metadata, such as a Merkle tree for commitments.
    pub data: Arc<PcsProverData<SC>>,
}

/// Key generation data containing both verifier and prover preprocessed data.
#[derive(Default)]
pub struct PrepKeygenData<SC: StarkGenericConfig> {
    /// Verifier-side preprocessed commitment data.
    pub verifier_data: Option<VerifierPrepData<Com<SC>>>,
    /// Prover-side preprocessed trace data.
    pub prover_data: Option<ProverPrepData<SC>>,
}

impl<SC: StarkGenericConfig> PrepKeygenData<SC> {
    /// Creates keygen data from a polynomial commitment scheme and an AIR.
    pub fn from_pcs_and_air(pcs: &SC::Pcs, air: &dyn DynamicAir<SC>) -> Self {
        air.preprocessed_trace().map_or_else(
            || Self {
                prover_data: None,
                verifier_data: None,
            },
            |trace| {
                let domain = pcs.natural_domain_for_degree(trace.height());
                let (commit, data) = pcs.commit(vec![(domain, trace.clone())]);
                Self {
                    prover_data: Some(ProverPrepData {
                        trace: Arc::new(trace),
                        data: Arc::new(data),
                    }),
                    verifier_data: Some(commit.into()),
                }
            },
        )
    }

    /// Returns the width of the preprocessed trace, if available.
    pub fn width(&self) -> Option<usize> {
        self.prover_data.as_ref().map(|d| d.trace.width())
    }
}
