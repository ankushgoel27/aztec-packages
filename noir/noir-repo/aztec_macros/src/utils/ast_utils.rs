use noirc_errors::{Span, Spanned};
use noirc_frontend::ast::{
    BinaryOpKind, CallExpression, CastExpression, Expression, ExpressionKind, FunctionReturnType,
    Ident, IndexExpression, InfixExpression, Lambda, MemberAccessExpression, MethodCallExpression,
    NoirTraitImpl, Path, PathSegment, Pattern, PrefixExpression, Statement, StatementKind,
    TraitImplItemKind, UnaryOp, UnresolvedType, UnresolvedTypeData,
};
use noirc_frontend::token::SecondaryAttribute;

//
//             Helper macros for creating noir ast nodes
//
pub fn ident(name: &str) -> Ident {
    Ident::new(name.to_string(), Span::default())
}

pub fn ident_path(name: &str) -> Path {
    Path::from_ident(ident(name))
}

pub fn path_segment(name: &str) -> PathSegment {
    PathSegment::from(ident(name))
}

pub fn path(ident: Ident) -> Path {
    Path::from_ident(ident)
}

pub fn expression(kind: ExpressionKind) -> Expression {
    Expression::new(kind, Span::default())
}

pub fn variable(name: &str) -> Expression {
    expression(ExpressionKind::Variable(ident_path(name)))
}

pub fn variable_ident(identifier: Ident) -> Expression {
    expression(ExpressionKind::Variable(path(identifier)))
}

pub fn variable_path(path: Path) -> Expression {
    expression(ExpressionKind::Variable(path))
}

pub fn method_call(
    object: Expression,
    method_name: &str,
    arguments: Vec<Expression>,
) -> Expression {
    expression(ExpressionKind::MethodCall(Box::new(MethodCallExpression {
        object,
        method_name: ident(method_name),
        arguments,
        is_macro_call: false,
        generics: None,
    })))
}

pub fn call(func: Expression, arguments: Vec<Expression>) -> Expression {
    expression(ExpressionKind::Call(Box::new(CallExpression {
        func: Box::new(func),
        is_macro_call: false,
        arguments,
    })))
}

pub fn pattern(name: &str) -> Pattern {
    Pattern::Identifier(ident(name))
}

pub fn mutable(name: &str) -> Pattern {
    Pattern::Mutable(Box::new(pattern(name)), Span::default(), true)
}

pub fn mutable_assignment(name: &str, assigned_to: Expression) -> Statement {
    make_statement(StatementKind::new_let(
        mutable(name),
        make_type(UnresolvedTypeData::Unspecified),
        assigned_to,
    ))
}

pub fn mutable_reference(variable_name: &str) -> Expression {
    expression(ExpressionKind::Prefix(Box::new(PrefixExpression {
        operator: UnaryOp::MutableReference,
        rhs: variable(variable_name),
    })))
}

pub fn assignment(name: &str, assigned_to: Expression) -> Statement {
    assignment_with_type(name, UnresolvedTypeData::Unspecified, assigned_to)
}

pub fn assignment_with_type(
    name: &str,
    typ: UnresolvedTypeData,
    assigned_to: Expression,
) -> Statement {
    make_statement(StatementKind::new_let(pattern(name), make_type(typ), assigned_to))
}

pub fn return_type(path: Path) -> FunctionReturnType {
    let ty = make_type(UnresolvedTypeData::Named(path, Default::default(), true));
    FunctionReturnType::Ty(ty)
}

pub fn lambda(parameters: Vec<(Pattern, UnresolvedType)>, body: Expression) -> Expression {
    expression(ExpressionKind::Lambda(Box::new(Lambda {
        parameters,
        return_type: UnresolvedType { typ: UnresolvedTypeData::Unspecified, span: Span::default() },
        body,
    })))
}

pub fn make_eq(lhs: Expression, rhs: Expression) -> Expression {
    expression(ExpressionKind::Infix(Box::new(InfixExpression {
        lhs,
        rhs,
        operator: Spanned::from(Span::default(), BinaryOpKind::Equal),
    })))
}

pub fn make_statement(kind: StatementKind) -> Statement {
    Statement { span: Span::default(), kind }
}

pub fn member_access(lhs: Expression, member: &str) -> Expression {
    expression(ExpressionKind::MemberAccess(Box::new(MemberAccessExpression {
        lhs,
        rhs: ident(member),
    })))
}

#[macro_export]
macro_rules! chained_path {
    ( $base:expr ) => {
        {
            ident_path($base)
        }
    };
    ( $base:expr $(, $tail:expr)* ) => {
        {
            let mut base_path = ident_path($base);
            $(
                base_path.segments.push(path_segment($tail));
            )*
            base_path
        }
    }
}

#[macro_export]
macro_rules! chained_dep {
    ( $base:expr $(, $tail:expr)* ) => {
        {
            let mut base_path = ident_path($base);
            base_path.kind = PathKind::Plain;
            $(
                base_path.segments.push(path_segment($tail));
            )*
            base_path
        }
    }
}

pub fn cast(lhs: Expression, ty: UnresolvedTypeData) -> Expression {
    expression(ExpressionKind::Cast(Box::new(CastExpression { lhs, r#type: make_type(ty) })))
}

pub fn make_type(typ: UnresolvedTypeData) -> UnresolvedType {
    UnresolvedType { typ, span: Span::default() }
}

pub fn index_array(array: Ident, index: &str) -> Expression {
    expression(ExpressionKind::Index(Box::new(IndexExpression {
        collection: variable_path(path(array)),
        index: variable(index),
    })))
}

pub fn check_trait_method_implemented(trait_impl: &NoirTraitImpl, method_name: &str) -> bool {
    trait_impl.items.iter().any(|item| match &item.item.kind {
        TraitImplItemKind::Function(func) => func.def.name.0.contents == method_name,
        _ => false,
    })
}

/// Checks if an attribute is a custom attribute with a specific name
pub fn is_custom_attribute(attr: &SecondaryAttribute, attribute_name: &str) -> bool {
    if let SecondaryAttribute::Custom(custom_attribute) = attr {
        custom_attribute.contents.as_str() == attribute_name
    } else {
        false
    }
}
