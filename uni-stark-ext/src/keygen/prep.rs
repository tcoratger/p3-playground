use p3_matrix::Matrix;

use super::types::{ProverOnlySinglePreprocessedData, VerifierSinglePreprocessedData};
use crate::{Com, StarkGenericConfig};

pub(super) struct PrepKeygenData<SC: StarkGenericConfig> {
    pub verifier_data: Option<VerifierSinglePreprocessedData<Com<SC>>>,
    pub prover_data: Option<ProverOnlySinglePreprocessedData<SC>>,
}

impl<SC: StarkGenericConfig> PrepKeygenData<SC> {
    pub fn width(&self) -> Option<usize> {
        self.prover_data.as_ref().map(|d| d.trace.width())
    }
}
