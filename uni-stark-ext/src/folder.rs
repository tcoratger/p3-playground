use alloc::vec::Vec;
use core::iter::{Skip, Take};
use core::ops::{Deref, Range};

use p3_air::{AirBuilder, AirBuilderWithPublicValues};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;

use crate::{
    Interaction, InteractionAirBuilder, InteractionType, PackedChallenge, PackedVal,
    StarkGenericConfig, Val,
};

#[derive(Debug)]
pub struct ProverConstraintFolder<'a, SC: StarkGenericConfig> {
    pub main: RowMajorMatrixView<'a, PackedVal<SC>>,
    pub public_values: &'a Vec<Val<SC>>,
    pub is_first_row: PackedVal<SC>,
    pub is_last_row: PackedVal<SC>,
    pub is_transition: PackedVal<SC>,
    pub alpha_powers: &'a [SC::Challenge],
    pub accumulator: PackedChallenge<SC>,
    pub constraint_index: usize,
}

type ViewPair<'a, T> = VerticalPair<RowMajorMatrixView<'a, T>, RowMajorMatrixView<'a, T>>;

#[derive(Debug)]
pub struct VerifierConstraintFolder<'a, SC: StarkGenericConfig> {
    pub main: ViewPair<'a, SC::Challenge>,
    pub public_values: &'a Vec<Val<SC>>,
    pub is_first_row: SC::Challenge,
    pub is_last_row: SC::Challenge,
    pub is_transition: SC::Challenge,
    pub alpha: SC::Challenge,
    pub accumulator: SC::Challenge,
}

impl<'a, SC: StarkGenericConfig> AirBuilder for ProverConstraintFolder<'a, SC> {
    type F = Val<SC>;
    type Expr = PackedVal<SC>;
    type Var = PackedVal<SC>;
    type M = RowMajorMatrixView<'a, PackedVal<SC>>;

    #[inline]
    fn main(&self) -> Self::M {
        self.main
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    #[inline]
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x: PackedVal<SC> = x.into();
        let alpha_power = self.alpha_powers[self.constraint_index];
        self.accumulator += Into::<PackedChallenge<SC>>::into(alpha_power) * x;
        self.constraint_index += 1;
    }
}

impl<SC: StarkGenericConfig> AirBuilderWithPublicValues for ProverConstraintFolder<'_, SC> {
    type PublicVar = Self::F;

    #[inline]
    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}

impl<SC: StarkGenericConfig> InteractionAirBuilder for ProverConstraintFolder<'_, SC> {
    fn push_interaction(
        &mut self,
        _bus_index: usize,
        _fields: impl IntoIterator<Item: Into<Self::Expr>>,
        _count: impl Into<Self::Expr>,
        _interaction_type: InteractionType,
    ) {
    }

    fn interactions(&self) -> &[Interaction<Self::Expr>] {
        &[]
    }
}

impl<'a, SC: StarkGenericConfig> AirBuilder for VerifierConstraintFolder<'a, SC> {
    type F = Val<SC>;
    type Expr = SC::Challenge;
    type Var = SC::Challenge;
    type M = ViewPair<'a, SC::Challenge>;

    fn main(&self) -> Self::M {
        self.main
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x: SC::Challenge = x.into();
        self.accumulator *= self.alpha;
        self.accumulator += x;
    }
}

impl<SC: StarkGenericConfig> AirBuilderWithPublicValues for VerifierConstraintFolder<'_, SC> {
    type PublicVar = Self::F;

    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}

impl<SC: StarkGenericConfig> InteractionAirBuilder for VerifierConstraintFolder<'_, SC> {
    fn push_interaction(
        &mut self,
        _bus_index: usize,
        _fields: impl IntoIterator<Item: Into<Self::Expr>>,
        _count: impl Into<Self::Expr>,
        _interaction_type: InteractionType,
    ) {
    }

    fn interactions(&self) -> &[Interaction<Self::Expr>] {
        &[]
    }
}

pub struct SubMatrixRowSlice<R> {
    inner: R,
    range: Range<usize>,
}

impl<R> SubMatrixRowSlice<R> {
    #[inline]
    pub const fn new(inner: R, range: Range<usize>) -> Self {
        Self { inner, range }
    }
}

impl<T, R: Deref<Target = [T]>> Deref for SubMatrixRowSlice<R> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner[self.range.clone()]
    }
}

pub struct SubMatrix<M> {
    inner: M,
    range: Range<usize>,
}

impl<M> SubMatrix<M> {
    #[inline]
    pub const fn new(inner: M, range: Range<usize>) -> Self {
        Self { inner, range }
    }
}

impl<M: Matrix<T>, T: Send + Sync> Matrix<T> for SubMatrix<M> {
    type Row<'a>
        = Take<Skip<M::Row<'a>>>
    where
        Self: 'a;

    #[inline]
    fn row(&self, r: usize) -> Self::Row<'_> {
        self.inner
            .row(r)
            .skip(self.range.start)
            .take(self.range.end)
    }

    #[inline]
    fn row_slice(&self, r: usize) -> impl Deref<Target = [T]> {
        SubMatrixRowSlice::new(self.inner.row_slice(r), self.range.clone())
    }

    #[inline]
    fn width(&self) -> usize {
        self.range.len()
    }

    #[inline]
    fn height(&self) -> usize {
        self.inner.height()
    }
}

pub struct SubAirBuilder<'a, AB> {
    inner: &'a mut AB,
    range: Range<usize>,
}

impl<'a, AB> SubAirBuilder<'a, AB> {
    #[inline]
    pub fn new(inner: &'a mut AB, range: Range<usize>) -> Self {
        Self { inner, range }
    }
}

impl<AB: AirBuilder> AirBuilder for SubAirBuilder<'_, AB> {
    type F = AB::F;
    type Expr = AB::Expr;
    type Var = AB::Var;
    type M = SubMatrix<AB::M>;

    #[inline]
    fn main(&self) -> Self::M {
        SubMatrix::new(self.inner.main(), self.range.clone())
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        self.inner.is_first_row()
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        self.inner.is_last_row()
    }

    #[inline]
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        self.inner.is_transition_window(size)
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.inner.assert_zero(x)
    }
}

impl<AB: AirBuilderWithPublicValues> AirBuilderWithPublicValues for SubAirBuilder<'_, AB> {
    type PublicVar = AB::PublicVar;

    #[inline]
    fn public_values(&self) -> &[Self::PublicVar] {
        self.inner.public_values()
    }
}
