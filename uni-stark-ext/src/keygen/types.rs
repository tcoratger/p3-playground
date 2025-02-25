use alloc::sync::Arc;
use alloc::vec::Vec;

use p3_matrix::dense::RowMajorMatrix;
use serde::{Deserialize, Serialize};

use crate::{PcsProverData, StarkGenericConfig, Val};

/// Verifier data for preprocessed trace for a single AIR.
///
/// Currently assumes each AIR has it's own preprocessed commitment
#[derive(Clone, Serialize, Deserialize)]
pub struct VerifierSinglePreprocessedData<Com> {
    /// Commitment to the preprocessed trace.
    pub commit: Com,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct StarkVerifyingParams {
    /// Trace sub-matrix widths
    pub width: TraceWidth,
    /// Number of public values for this STARK only
    pub num_public_values: usize,
    /// Number of values to expose to verifier in each trace challenge phase
    pub num_exposed_values_after_challenge: Vec<usize>,
    /// For only this RAP, how many challenges are needed in each trace challenge phase
    pub num_challenges_to_sample: Vec<usize>,
}

/// Widths of different parts of trace matrix
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceWidth {
    pub preprocessed: Option<usize>,
    pub cached_mains: Vec<usize>,
    pub common_main: usize,
    /// Width counted by extension field elements, _not_ base field elements
    pub after_challenge: Vec<usize>,
}

/// Verifying key for a single STARK (corresponding to single AIR matrix)
#[derive(Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct StarkVerifyingKey<Com> {
    /// Preprocessed trace data, if any
    pub preprocessed_data: Option<VerifierSinglePreprocessedData<Com>>,
    /// Parameters of the STARK
    pub params: StarkVerifyingParams,
    /// The factor to multiple the trace degree by to get the degree of the quotient polynomial. Determined from the max constraint degree of the AIR constraints.
    /// This is equivalently the number of chunks the quotient polynomial is split into.
    pub quotient_degree: u8,
}

/// Prover only data for preprocessed trace for a single AIR.
/// Currently assumes each AIR has it's own preprocessed commitment
pub struct ProverOnlySinglePreprocessedData<SC: StarkGenericConfig> {
    /// Preprocessed trace matrix.
    pub trace: Arc<RowMajorMatrix<Val<SC>>>,
    /// Prover data, such as a Merkle tree, for the trace commitment.
    pub data: Arc<PcsProverData<SC>>,
}
