use alloc::sync::Arc;
use alloc::vec;

use p3_commit::Pcs;
use p3_matrix::Matrix;

use super::prep::PrepKeygenData;
use super::types::{ProverOnlySinglePreprocessedData, VerifierSinglePreprocessedData};
use crate::{AnyRap, StarkGenericConfig, get_max_constraint_degree};

pub struct AirKeygenBuilder<SC: StarkGenericConfig> {
    air: Arc<dyn AnyRap<SC>>,
    prep_keygen_data: PrepKeygenData<SC>,
}

impl<SC: StarkGenericConfig> AirKeygenBuilder<SC> {
    pub fn new(pcs: &SC::Pcs, air: Arc<dyn AnyRap<SC>>) -> Self {
        let preprocessed_trace = air.preprocessed_trace();
        let vpdata_opt = preprocessed_trace.map(|trace| {
            let domain = pcs.natural_domain_for_degree(trace.height());
            let (commit, data) = pcs.commit(vec![(domain, trace.clone())]);
            let vdata = VerifierSinglePreprocessedData { commit };
            let pdata = ProverOnlySinglePreprocessedData::<SC> {
                trace: Arc::new(trace),
                data: Arc::new(data),
            };
            (vdata, pdata)
        });
        let prep_keygen_data = if let Some((vdata, pdata)) = vpdata_opt {
            PrepKeygenData {
                prover_data: Some(pdata),
                verifier_data: Some(vdata),
            }
        } else {
            PrepKeygenData {
                prover_data: None,
                verifier_data: None,
            }
        };

        AirKeygenBuilder {
            air,
            prep_keygen_data,
        }
    }

    fn max_constraint_degree(&self) -> usize {
        get_max_constraint_degree(
            self.air.as_ref(),
            self.prep_keygen_data.width().unwrap_or(0),
            self.air.num_public_values(),
        )
    }
}
