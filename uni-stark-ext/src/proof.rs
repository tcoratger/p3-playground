use alloc::vec::Vec;

use p3_commit::Pcs;
use serde::{Deserialize, Serialize};

use crate::StarkGenericConfig;

type Com<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Commitment;
type PcsProof<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Proof;

#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Proof<SC: StarkGenericConfig> {
    pub(crate) commitments: Commitments<Com<SC>>,
    pub(crate) per_air: Vec<ProofPerAir<SC>>,
    pub(crate) opening_proof: PcsProof<SC>,
}

#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProofPerAir<SC: StarkGenericConfig> {
    pub(crate) log_degree: usize,
    pub(crate) log_up_sum: Option<SC::Challenge>,
    pub(crate) opened_values: OpenedValues<SC::Challenge>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Commitments<Com> {
    pub(crate) main: Com,
    pub(crate) log_up_chunks: Option<Com>,
    pub(crate) quotient_chunks: Com,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenedValues<Challenge> {
    pub(crate) main_local: Vec<Challenge>,
    pub(crate) main_next: Vec<Challenge>,
    pub(crate) log_up_local: Vec<Challenge>,
    pub(crate) log_up_next: Vec<Challenge>,
    pub(crate) quotient_chunks: Vec<Vec<Challenge>>,
}
