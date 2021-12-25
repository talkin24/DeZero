# -*- coding: utf-8 -*-

# step23.py에서 step32.py까지는 simple_core를 이용
is_simple_core = False

if is_simple_core:
    from dezero.core_simple import Variable, as_array
    from dezero.core_simple import Function
    from dezero.core_simple import using_config
    from dezero.core_simple import no_grad
    from dezero.core_simple import as_array
    from dezero.core_simple import as_variable
    from dezero.core_simple import setup_variable

else:
    pass

setup_variable()
