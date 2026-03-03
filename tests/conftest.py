import jax

jax.config.update("jax_disable_jit", True)
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_compilation_cache", True)
jax.config.update("jax_compilation_cache_dir", ".jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
