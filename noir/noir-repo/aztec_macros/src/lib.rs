mod transforms;
mod utils;

use noirc_errors::Location;
use transforms::{
    compute_note_hash_and_optionally_a_nullifier::inject_compute_note_hash_and_optionally_a_nullifier,
    contract_interface::{
        generate_contract_interface, stub_function, update_fn_signatures_in_contract_interface,
    },
    events::{generate_event_impls, transform_event_abi},
    functions::{
        check_for_public_args, export_fn_abi, transform_function, transform_unconstrained,
    },
    note_interface::{generate_note_interface_impl, inject_note_exports},
    storage::{
        assign_storage_slots, check_for_storage_definition, check_for_storage_implementation,
        generate_storage_implementation, generate_storage_layout, inject_context_in_storage,
    },
};

use noirc_frontend::macros_api::{
    CrateId, FileId, HirContext, MacroError, MacroProcessor, SortedModule, Span,
};

use utils::{
    ast_utils::is_custom_attribute,
    checks::{check_for_aztec_dependency, has_aztec_dependency},
    constants::MAX_CONTRACT_PRIVATE_FUNCTIONS,
    errors::AztecMacroError,
};
pub struct AztecMacro;

impl MacroProcessor for AztecMacro {
    fn process_untyped_ast(
        &self,
        ast: SortedModule,
        crate_id: &CrateId,
        file_id: FileId,
        context: &HirContext,
    ) -> Result<SortedModule, (MacroError, FileId)> {
        transform(ast, crate_id, file_id, context)
    }

    fn process_typed_ast(
        &self,
        crate_id: &CrateId,
        context: &mut HirContext,
    ) -> Result<(), (MacroError, FileId)> {
        transform_hir(crate_id, context).map_err(|(err, file_id)| (err.into(), file_id))
    }
}

//
//                    Create AST Nodes for Aztec
//

/// Traverses every function in the ast, calling `transform_function` which
/// determines if further processing is required
fn transform(
    mut ast: SortedModule,
    crate_id: &CrateId,
    file_id: FileId,
    context: &HirContext,
) -> Result<SortedModule, (MacroError, FileId)> {
    let empty_spans = context.def_interner.is_in_lsp_mode();

    // Usage -> mut ast -> aztec_library::transform(&mut ast)
    // Covers all functions in the ast
    for submodule in
        ast.submodules.iter_mut().map(|m| &mut m.item).filter(|submodule| submodule.is_contract)
    {
        if transform_module(
            &file_id,
            &mut submodule.contents,
            submodule.name.0.contents.as_str(),
            empty_spans,
        )
        .map_err(|err| (err.into(), file_id))?
        {
            check_for_aztec_dependency(crate_id, context)?;
        }
    }

    generate_event_impls(&mut ast, empty_spans).map_err(|err| (err.into(), file_id))?;
    generate_note_interface_impl(&mut ast, empty_spans).map_err(|err| (err.into(), file_id))?;

    Ok(ast)
}

/// Determines if ast nodes are annotated with aztec attributes.
/// For annotated functions it calls the `transform` function which will perform the required transformations.
/// Returns true if an annotated node is found, false otherwise
fn transform_module(
    file_id: &FileId,
    module: &mut SortedModule,
    module_name: &str,
    empty_spans: bool,
) -> Result<bool, AztecMacroError> {
    let mut has_transformed_module = false;

    // Check for a user defined storage struct

    let maybe_storage_struct_name = check_for_storage_definition(module)?;

    let storage_defined = maybe_storage_struct_name.is_some();

    if let Some(ref storage_struct_name) = maybe_storage_struct_name {
        inject_context_in_storage(module)?;
        if !check_for_storage_implementation(module, storage_struct_name) {
            generate_storage_implementation(module, storage_struct_name)?;
        }
        generate_storage_layout(module, storage_struct_name.clone(), module_name, empty_spans)?;
    }

    let has_initializer = module.functions.iter().any(|func| {
        func.item
            .def
            .attributes
            .secondary
            .iter()
            .any(|attr| is_custom_attribute(attr, "aztec(initializer)"))
    });

    let mut stubs: Vec<_> = vec![];

    for func in module.functions.iter_mut() {
        let func = &mut func.item;
        let mut is_private = false;
        let mut is_public = false;
        let mut is_initializer = false;
        let mut is_internal = false;
        let mut insert_init_check = has_initializer;
        let mut is_static = false;

        for secondary_attribute in func.def.attributes.secondary.clone() {
            if is_custom_attribute(&secondary_attribute, "aztec(private)") {
                is_private = true;
            } else if is_custom_attribute(&secondary_attribute, "aztec(initializer)") {
                is_initializer = true;
                insert_init_check = false;
            } else if is_custom_attribute(&secondary_attribute, "aztec(noinitcheck)") {
                insert_init_check = false;
            } else if is_custom_attribute(&secondary_attribute, "aztec(internal)") {
                is_internal = true;
            } else if is_custom_attribute(&secondary_attribute, "aztec(public)") {
                is_public = true;
            }
            if is_custom_attribute(&secondary_attribute, "aztec(view)") {
                is_static = true;
            }
        }

        // Apply transformations to the function based on collected attributes
        if is_private || is_public {
            let fn_type = if is_private { "Private" } else { "Public" };
            let stub_src = stub_function(fn_type, func, is_static);
            stubs.push((stub_src, Location { file: *file_id, span: func.name_ident().span() }));

            export_fn_abi(&mut module.types, func, empty_spans)?;
            transform_function(
                fn_type,
                func,
                maybe_storage_struct_name.clone(),
                is_initializer,
                insert_init_check,
                is_internal,
                is_static,
            )?;
            has_transformed_module = true;
        } else if storage_defined && func.def.is_unconstrained {
            transform_unconstrained(func, maybe_storage_struct_name.clone().unwrap());
            has_transformed_module = true;
        }
    }

    if has_transformed_module {
        // We only want to run these checks if the macro processor has found the module to be an Aztec contract.

        let private_functions: Vec<_> = module
            .functions
            .iter()
            .map(|t| &t.item)
            .filter(|func| {
                func.def
                    .attributes
                    .secondary
                    .iter()
                    .any(|attr| is_custom_attribute(attr, "aztec(private)"))
            })
            .collect();

        let public_functions: Vec<_> = module
            .functions
            .iter()
            .map(|func| &func.item)
            .filter(|func| {
                func.def
                    .attributes
                    .secondary
                    .iter()
                    .any(|attr| is_custom_attribute(attr, "aztec(public)"))
            })
            .collect();

        let private_function_count = private_functions.len();

        check_for_public_args(&private_functions)?;

        check_for_public_args(&public_functions)?;

        if private_function_count > MAX_CONTRACT_PRIVATE_FUNCTIONS {
            return Err(AztecMacroError::ContractHasTooManyPrivateFunctions {
                span: Span::default(),
            });
        }

        generate_contract_interface(module, module_name, &stubs, storage_defined, empty_spans)?;
    }

    Ok(has_transformed_module)
}

//
//                    Transform Hir Nodes for Aztec
//

/// Completes the Hir with data gathered from type resolution
fn transform_hir(
    crate_id: &CrateId,
    context: &mut HirContext,
) -> Result<(), (AztecMacroError, FileId)> {
    if has_aztec_dependency(crate_id, context) {
        transform_event_abi(crate_id, context)?;
        inject_compute_note_hash_and_optionally_a_nullifier(crate_id, context)?;
        assign_storage_slots(crate_id, context)?;
        inject_note_exports(crate_id, context)?;
        update_fn_signatures_in_contract_interface(crate_id, context)
    } else {
        Ok(())
    }
}
