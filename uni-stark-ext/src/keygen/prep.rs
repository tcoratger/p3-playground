use alloc::vec;

use p3_commit::Pcs;
use p3_matrix::Matrix;
use serde::{Deserialize, Serialize};

use crate::{Com, DynamicAir, StarkGenericConfig};

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

/// Key generation data containing both verifier and prover preprocessed data.
#[derive(Default)]
pub struct PrepKeygenData<SC: StarkGenericConfig> {
    /// Verifier-side preprocessed commitment data.
    pub verifier_data: Option<VerifierPrepData<Com<SC>>>,
}

impl<SC: StarkGenericConfig> PrepKeygenData<SC> {
    /// Creates keygen data from a polynomial commitment scheme and an AIR.
    pub fn from_pcs_and_air(pcs: &SC::Pcs, air: &dyn DynamicAir<SC>) -> Self {
        air.preprocessed_trace().map_or_else(
            || Self {
                verifier_data: None,
            },
            |trace| {
                let domain = pcs.natural_domain_for_degree(trace.height());
                let (commit, _) = pcs.commit(vec![(domain, trace.clone())]);
                Self {
                    verifier_data: Some(commit.into()),
                }
            },
        )
    }
}
