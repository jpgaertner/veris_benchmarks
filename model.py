from veros import veros_kernel, veros_routine, KernelOutput
from veros.core.operators import numpy as npx

from veris.area_mass import SeaIceMass, AreaWS
from veris.dynsolver import WindForcingXY, IceVelocities
from veris.dynamics_routines import SeaIceStrength
from veris.ocean_stress import OceanStressUV
from veris.advection import Advection
from veris.clean_up import clean_up_advection, ridging
from veris.fill_overlap import fill_overlap
from veris.growth import Growth


@veros_routine
def dyn_model(state):
    vs = state.variables

    # calculate sea ice mass centered around c-, u-, and v-points
    vs.SeaIceMassC, vs.SeaIceMassU, vs.SeaIceMassV = SeaIceMass(state)

    # calculate sea ice cover fraction centered around u- and v-points
    vs.AreaW, vs.AreaS = AreaWS(state)

    # calculate surface forcing due to wind
    vs.WindForcingX, vs.WindForcingY = WindForcingXY(state)

    # calculate ice strength
    vs.SeaIceStrength = SeaIceStrength(state)

    # calculate ice velocities
    vs.uIce, vs.vIce, vs.sigma1, vs.sigma2, vs.sigma12 = IceVelocities(state)

    # calculate stresses on ocean surface
    vs.OceanStressU, vs.OceanStressV = OceanStressUV(state)

    # calculate change in sea ice fields due to advection
    vs.hIceMean, vs.hSnowMean, vs.Area = Advection(state)

    # correct overshoots and other pathological cases after advection
    (
        vs.hIceMean,
        vs.hSnowMean,
        vs.Area,
        vs.TSurf,
        vs.os_hIceMean,
        vs.os_hSnowMean,
    ) = clean_up_advection(state)

    # cut off ice cover fraction at 1 after advection
    vs.Area = ridging(state)

    # fill overlaps
    vs.hIceMean = fill_overlap(state,vs.hIceMean)
    vs.hSnowMean = fill_overlap(state,vs.hSnowMean)
    vs.Area = fill_overlap(state,vs.Area)

@veros_kernel
def output_growth(state):
    (
        hIceMean,
        hSnowMean,
        Area,
        TSurf,
        EmPmR,
        forc_salt_surface,
        Qsw,
        Qnet,
        SeaIceLoad,
        IcePenetSW,
        recip_hIceMean,
    ) = Growth(state)

    return KernelOutput(hIceMean = hIceMean,
                        hSnowMean = hSnowMean,
                        Area = Area,
                        TSurf = TSurf,
                        EmPmR = EmPmR,
                        # forc_salt_surface is just an output for veros
                        # which imports the Growth function, not output_growth.
                        # forc_salt_surface is not defined in state.variables in veris
                        Qsw = Qsw,
                        Qnet = Qnet,
                        SeaIceLoad = SeaIceLoad,
                        IcePenetSW = IcePenetSW,
                        recip_hIceMean = recip_hIceMean)

@veros_routine
def update_growth(state):
    state.variables.update(output_growth(state))
