use alloc::vec;
use alloc::vec::Vec;

use p3_air::{Air, BaseAir, BaseAirWithPublicValues, PermutationAirBuilder};

use crate::{DebugConstraintBuilder, StarkGenericConfig, SymbolicAirBuilder, Val};

/// RAP trait for all-purpose dynamic dispatch use.
pub trait AnyRap<SC: StarkGenericConfig>:
    Air<SymbolicAirBuilder<Val<SC>>>
    + BaseAirWithPublicValues<Val<SC>>
    + for<'a> Rap<DebugConstraintBuilder<'a, SC>>
    + PartitionedBaseAir<Val<SC>>
    + Send
    + Sync
{
}

pub trait Rap<AB>: Sync
where
    AB: PermutationAirBuilder,
{
    fn eval(&self, builder: &mut AB);
}

/// An AIR with 1 or more main trace partitions.
pub trait PartitionedBaseAir<F>: BaseAir<F> {
    /// By default, an AIR has no cached main trace.
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![]
    }
    /// By default, an AIR has only one private main trace.
    fn common_main_width(&self) -> usize {
        self.width()
    }
}
