import sys
from benchmark_base import benchmark_cli
from time import perf_counter
from veros import logger


path_forcing_fields = '   '

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

        deltat = 600
        input_settings = {
            'deltatTherm'       : deltat,
            'recip_deltatTherm' : 1 / deltat,
            'deltatDyn'         : deltat,
            'recip_deltatDyn'   : 1 / deltat,
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
            var = exchange_overlap(var, dims, False) #(settings.enable_cyclic_x=False)

            input_data[variable_name] = var

        with state.variables.unlock():
            for variable_name in file.keys():
                setattr(state.variables, variable_name, input_data[variable_name])
                
        return state, input_data

@benchmark_cli
def main(pyom2_lib, timesteps, size):
    from veros.core.operators import numpy as npx
    # the model has to be imported here bc it includes import numpy from veros
    # and "Runtime settings cannot be modified after import of core modules"
    from model import dyn_model

    infile = f'{path_forcing_fields}/initial_fields_img.h5'
    
    state, input_data = get_state(infile)
    vs = state.variables
    
    # use this for saving the ice field at every time step
    #ice, snow, area, uice, vice = [[] for i in range(5)]
    for i in range(timesteps):

        #if i%10==0:
        #ice += vs.hIceMean,
        #snow += vs.hSnowMean,
        #area += vs.Area,
        #uice += vs.uIce,
        #vice += vs.vIce,
        
        start = perf_counter()
        dyn_model(state)
        end = perf_counter()


        logger.debug(f"Time step took {end - start}s")
        
    #npx.save('image/veris_output', [ice, snow, area, uice, vice])
    npx.save('veris_output', [vs.hIceMean, vs.hSnowMean, vs.Area, vs.uIce, vs.vIce,
                                   vs.uOcean, vs.vOcean, vs.uWind, vs.vWind])
        
if __name__ == "__main__":
    main()
