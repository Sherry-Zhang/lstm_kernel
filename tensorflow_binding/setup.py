import os
import setuptools
from setuptools.command.build_ext import build_ext as orig_build_ext
import tensorflow as tf

tf_inc = tf.sysconfig.get_include()
tf_lib = tf.sysconfig.get_lib()
rnn_inc = os.path.realpath("../include")
rnn_lib = os.path.realpath("../build")

inc_dirs = [tf_inc, tf_inc + '/external/nsync/public', rnn_inc]
lib_dirs = [tf_lib, rnn_lib]

srcs = ['src/lstmop.cpp']

compile_args = ['-std=c++11', '-fPIC']
compile_args += ['-shared', '-O2']

ext = setuptools.Extension(
    "intel_lstm.kernels",
    language='c++',
    sources=srcs,
    include_dirs=inc_dirs,
    library_dirs=lib_dirs,
    libraries=['lstm', 'tensorflow_framework'],
    runtime_library_dirs=[rnn_lib],
    extra_compile_args=compile_args
)

class build_tf_ext(orig_build_ext):
    def build_extensions(self):
        self.compiler.compiler_so.remove('-Wstrict-prototypes')
        orig_build_ext.build_extensions(self)

setuptools.setup(
    name="intel_lstm",
    version="0.1",
    description="intelrnn lstm inference",
    packages=["intel_lstm"],
    ext_modules=[ext],
    cmdclass={'build_ext': build_tf_ext}
)
