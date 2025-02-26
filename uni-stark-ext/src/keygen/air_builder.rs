use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use p3_air::{Air, BaseAir};

use super::prep::{PrepKeygenData, ProverPrepData, VerifierPrepData};
use super::verifying_params::StarkVerifyingParams;
use crate::trace_width::TraceWidth;
use crate::{
    Com, DynamicAir, StarkGenericConfig, SymbolicAirBuilder, SymbolicExpression, Val,
    get_log_quotient_degree, get_max_constraint_degree,
};

/// A verifying key for a single STARK instance.
#[derive(Clone)]
#[repr(C)]
pub struct StarkVerifyingKey<Val, Com> {
    /// Preprocessed data for the verifier, if available.
    pub preprocessed_data: Option<VerifierPrepData<Com>>,
    /// Parameters describing the STARK verification process.
    pub params: StarkVerifyingParams,
    /// The symbolic constraints defining the AIR.
    pub symbolic_constraints: Vec<SymbolicExpression<Val>>,
    /// The factor used to compute the quotient polynomial degree.
    pub quotient_degree: u8,
}

/// A proving key for a single STARK instance.
pub struct StarkProvingKey<SC: StarkGenericConfig> {
    /// The AIR's name for identification.
    pub air_name: String,
    /// The corresponding verifying key.
    pub vk: StarkVerifyingKey<Val<SC>, Com<SC>>,
    /// Prover-specific preprocessed trace data.
    pub preprocessed_data: Option<ProverPrepData<SC>>,
}

/// A builder for generating STARK proving and verifying keys.
pub struct AirKeygenBuilder<SC: StarkGenericConfig> {
    /// The AIR instance associated with this builder.
    air: Arc<dyn DynamicAir<SC>>,
    /// Preprocessed key generation data for the AIR.
    prep_keygen_data: PrepKeygenData<SC>,
    /// The width of the execution trace.
    trace_width: TraceWidth,
}

impl<SC: StarkGenericConfig> AirKeygenBuilder<SC> {
    /// Creates a new key generator for an AIR.
    pub fn new(pcs: &SC::Pcs, air: Arc<dyn DynamicAir<SC>>) -> Self {
        let prep_keygen_data = PrepKeygenData::from_pcs_and_air(pcs, air.as_ref());
        Self {
            trace_width: TraceWidth::from_prep_and_air(&prep_keygen_data, air.as_ref()),
            prep_keygen_data,
            air,
        }
    }

    /// Returns the maximum constraint degree for this AIR.
    #[inline(always)]
    pub(crate) fn max_constraint_degree(&self) -> usize {
        get_max_constraint_degree(
            self.air.as_ref(),
            self.prep_keygen_data.width().unwrap_or_default(),
            self.air.num_public_values(),
        )
    }

    /// Constructs a symbolic representation of the AIR for constraint generation.
    #[inline(always)]
    pub(crate) fn symbolic_builder(&self) -> SymbolicAirBuilder<Val<SC>> {
        let mut builder = SymbolicAirBuilder::new(
            self.trace_width.preprocessed.unwrap_or_default(),
            <dyn DynamicAir<SC> as BaseAir<Val<SC>>>::width(self.air.as_ref()),
            self.air.num_public_values(),
        );
        Air::eval(self.air.as_ref(), &mut builder);
        builder
    }

    /// Computes the quotient polynomial degree for this AIR.
    #[inline(always)]
    fn quotient_degree(&self) -> u8 {
        1 << get_log_quotient_degree(
            self.air.as_ref(),
            self.prep_keygen_data.width().unwrap_or_default(),
            self.air.num_public_values(),
        )
    }

    /// Consumes the builder and generates a proving key.
    pub(crate) fn generate_pk(self) -> StarkProvingKey<SC> {
        let symbolic_builder = self.symbolic_builder();

        let vk = StarkVerifyingKey {
            quotient_degree: self.quotient_degree(),
            preprocessed_data: self.prep_keygen_data.verifier_data,
            params: StarkVerifyingParams::from_width_and_sbuilder(
                self.trace_width,
                &symbolic_builder,
            ),
            symbolic_constraints: symbolic_builder.constraints(),
        };

        StarkProvingKey {
            air_name: self.air.name(),
            vk,
            preprocessed_data: self.prep_keygen_data.prover_data,
        }
    }
}
