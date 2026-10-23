"""Drop-in CPLEX logging helpers. They do NOT change solver parameters."""
from __future__ import annotations
import json
import math
import platform
from pathlib import Path
import importlib.metadata


def _write(obj, path):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')


def _clean(v):
    if isinstance(v,float) and not math.isfinite(v):return None
    if isinstance(v,(str,int,float,bool)) or v is None:return v
    return str(v)


def effective_configuration(model):
    """Record effective settings; do not change any parameter."""
    names = ("timelimit", "threads", "randomseed", "parallel",
             "mip.tolerances.mipgap", "mip.tolerances.absmipgap",
             "mip.tolerances.integrality")
    values = {}
    for name in names:
        parameter = model.parameters
        for part in name.split("."):
            parameter = getattr(parameter, part)
        values[name] = {"value": _clean(parameter.get()),
                       "default": _clean(parameter.default_value),
                       "is_default": bool(parameter.is_default())}
    return values


def before_solve(model, directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    versions={}
    for name in ('docplex','cplex'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    parameters=[]
    for p in model.parameters.generate_params():
        parameters.append({'name':p.qualified_name,'cplex_name':p.cpx_name,'value':_clean(p.get()),'default':_clean(p.default_value),'is_default':bool(p.is_default())})
    meta={'python':platform.python_version(),'packages':versions,'model_name':model.name,'parameters':parameters,
          'purpose':'Current execution record; does not recover historical run parameters.'}
    try:meta['runtime_cplex_version']=model.get_cplex().get_version()
    except Exception as exc:meta['runtime_cplex_version_error']=type(exc).__name__
    _write(meta,directory/'cplex_parameters.json')
    (directory/'nondefault_parameters.prm').write_text(model.parameters.export_prm_to_string(),encoding='utf8')
    (directory/'model.lp').write_text(model.export_as_lp_string(),encoding='utf8')
    return meta


def after_solve(model, solution, directory):
    details=model.solve_details
    fields=('status','status_code','time','best_bound','mip_relative_gap','nb_nodes_processed','nb_iterations')
    result={k:_clean(getattr(details,k,None)) for k in fields}
    result['objective_value']=_clean(solution.objective_value) if solution is not None else None
    result['has_solution']=solution is not None
    result['solve_status']=str(getattr(solution,'solve_status',None)) if solution is not None else None
    # An optimal solver status is reported separately from any exact-integer bound check.
    code=getattr(details,'status_code',None)
    result['solver_reported_optimal']=code in (101,102)
    result['optimality_note']='Code 102 is optimal within the configured MIP tolerances. Preserve bound and gap; do not label timeout incumbents exact.'
    _write(result,Path(directory)/'solve_details.json')
    if solution is not None:
        values={v.name:float(solution.get_value(v)) for v in model.iter_variables()}
        _write(values,Path(directory)/'solution_values.json')
    return result
