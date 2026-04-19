"""
Start the dental aligner server on Colab (stubs openenv-core if missing).
Usage: python start_server_colab.py
"""
import sys
import types

# Stub openenv if not installed
try:
    import openenv
except ImportError:
    openenv = types.ModuleType('openenv')
    openenv.core = types.ModuleType('openenv.core')
    openenv.core.env_server = types.ModuleType('openenv.core.env_server')

    class _Base:
        pass

    imod = types.ModuleType('openenv.core.env_server.interfaces')
    imod.Environment = _Base
    tmod = types.ModuleType('openenv.core.env_server.types')
    tmod.Action = _Base
    tmod.Observation = _Base
    tmod.State = _Base
    openenv.core.env_server.interfaces = imod
    openenv.core.env_server.types = tmod

    def _create_fastapi_app(env, **kwargs):
        from fastapi import FastAPI
        return FastAPI()

    openenv.core.env_server.create_fastapi_app = _create_fastapi_app

    for k, v in {
        'openenv': openenv,
        'openenv.core': openenv.core,
        'openenv.core.env_server': openenv.core.env_server,
        'openenv.core.env_server.interfaces': imod,
        'openenv.core.env_server.types': tmod,
    }.items():
        sys.modules[k] = v

# Now start the server
import uvicorn
uvicorn.run('server.app:app', host='0.0.0.0', port=7860)
