//! A minimal univariate STARK framework.

#![no_std]

extern crate alloc;

mod config;
mod folder;
mod keygen;
mod proof;
mod prover;
mod rap;
mod symbolic_builder;
mod symbolic_expression;
mod symbolic_variable;
mod vanishing_polynomial_coset;
mod verifier;

mod check_constraints;

pub use check_constraints::*;
pub use config::*;
pub use folder::*;
pub use keygen::*;
pub use proof::*;
pub use prover::*;
pub use rap::*;
pub use symbolic_builder::*;
pub use symbolic_expression::*;
pub use symbolic_variable::*;
pub use vanishing_polynomial_coset::*;
pub use verifier::*;
