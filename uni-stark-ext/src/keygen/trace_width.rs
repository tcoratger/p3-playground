use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use super::prep::PrepKeygenData;
use crate::{DynamicAir, StarkGenericConfig};

/// Describes the widths of different sections of a trace matrix.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceWidth {
    /// Width of the preprocessed trace, if applicable.
    pub preprocessed: Option<usize>,
    /// Widths of cached main trace segments.
    pub cached_mains: Vec<usize>,
    /// The common width of the main trace.
    pub common_main: usize,
    /// Width counted in extension field elements, not base field elements.
    pub after_challenge: Vec<usize>,
}

impl TraceWidth {
    /// Constructs `TraceWidth` from preprocessed data and an AIR.
    pub(crate) fn from_prep_and_air<SC>(
        prep_data: &PrepKeygenData<SC>,
        air: &dyn DynamicAir<SC>,
    ) -> Self
    where
        SC: StarkGenericConfig,
    {
        Self {
            preprocessed: prep_data.width(),
            cached_mains: air.cached_main_widths(),
            common_main: air.common_main_width(),
            after_challenge: Vec::new(),
        }
    }
}
