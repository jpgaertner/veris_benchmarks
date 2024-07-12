from benchmark_base import benchmark_cli
from time import perf_counter
from veros import logger
import numpy as np


def get_state(infile):
    import h5py
    from veros import runtime_settings
    from veros.core.operators import update, at, numpy as npx
    from veros.distributed import get_chunk_slices, exchange_overlap
    from veros.state import VerosState
    from veros.settings import Setting
    from veros.variables import get_shape
    from veris.variables import VARIABLES
    from veris.settings import SETTINGS
    

    with h5py.File(infile, 'r') as f:
        gridx = f['hIceMean'].shape[0]
        gridy = f['hIceMean'].shape[1]
 
    SETTINGS['nx'] = Setting(gridx-4, int, "Number of grid cells in x direction [ ]")
    SETTINGS['ny'] = Setting(gridy-4, int, "Number of grid cells in Y direction [ ]")

    state = VerosState(var_meta=VARIABLES, setting_meta=SETTINGS, dimensions={'xt':'nx', 'yt':'ny'})

    with h5py.File(infile, 'r') as file:

        input_settings = {
            'deltatTherm'       : 3600,
            'recip_deltatTherm' : 1 / 3600,
            'deltatDyn'         : 3600,
            'recip_deltatDyn'   : 1 / 3600,
            'gridcellWidth'     : 8000,
            'veros_fill'        : False,
            'useEVP'            : True,
            'useAdaptiveEVP'    : True,
            'useRelativeWind'   : False,
            'evpAlpha'          : 123456.7,
            'evpBeta'           : 123456.7,
            'nEVPsteps'         : 400
        }

        with state.settings.unlock():
            state.settings.update(input_settings)

        state.initialize_variables()

        dimensions = state.dimensions
        dims = ('xt', 'yt')
        local_shape = get_shape(dimensions, dims, local=True, include_ghosts=True)
        gidx, lidx = get_chunk_slices(dimensions['xt'], dimensions['yt'], dims, include_overlap=True)

        var_meta = state.var_meta
        input_data = {}

        for variable_name in file.keys():

            var = npx.empty(local_shape, dtype=file[variable_name].dtype)
            var = update(var, at[lidx], file[variable_name][gidx])
            print('test', npx.shape(var), npx.shape(dims))
            var = exchange_overlap(var, dims, False) #(settings.enable_cyclic_x=False)

            input_data[variable_name] = var

        with state.variables.unlock():
            for variable_name in file.keys():
                setattr(state.variables, variable_name, input_data[variable_name])
                
        return state, input_data

def increase_counter(counter):
    if 'counter' in locals():
        return counter+1
    else:
        return 0

@benchmark_cli
def main(pyom2_lib, timesteps, size):
    # the model has to be imported here bc it includes import numpy from veros
    # and "Runtime settings cannot be modified after import of core modules"
    from model import dyn_model 
    
    infile = 'initial_fields.h5'
    
    state, input_data = get_state(infile)
    vs = state.variables
    
    ice, snow, area, uice, vice = [[] for i in range(5)]
    for i in range(timesteps):
        start = perf_counter()
        dyn_model(state)
        end = perf_counter()
        ice += vs.hIceMean,
        snow += vs.hSnowMean,
        area += vs.Area,
        uice += vs.uIce,
        vice += vs.vIce,

        logger.debug(f"Time step took {end - start}s")
        
    np.save('/home/a/a270230/models/veris_benchmarks/veris_output', [ice, snow, area, uice, vice])
        
if __name__ == "__main__":
    main()
