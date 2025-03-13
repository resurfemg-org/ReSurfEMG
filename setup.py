#!/usr/bin/env python
import os
import sys
from glob import glob
import subprocess

from setuptools import Command, setup
import mock

MOCK_MODULES = ['adi']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()


project_dir = os.path.dirname(os.path.realpath(__file__))


class SphinxApiDoc(Command):

    description = 'run apidoc to generate documentation'

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        from sphinx.ext.apidoc import main

        src = os.path.join(project_dir, 'docs')
        special = (
            'index.rst',
            'developers.rst',
            'medical-professionals.rst',
        )

        for f in glob(os.path.join(src, '*.rst')):
            for end in special:
                if f.endswith(end):
                    os.utime(f, None)
                    break
            else:
                os.unlink(f)

        sys.exit(main([
            '-o', src,
            '-f',
            os.path.join(project_dir, 'resurfemg'),
            '--separate',
        ]))


class SphinxDoc(Command):

    description = 'generate documentation'

    user_options = [('wall', 'W', ('Warnings are errors'))]

    def initialize_options(self):
        self.wall = True

    def finalize_options(self):
        pass

    def run(self):
        from sphinx.util.console import nocolor
        from sphinx.util.docutils import docutils_namespace, patch_docutils
        from sphinx.application import Sphinx
        from sphinx.cmd.build import handle_exception

        name = 'resurfemg'
        try:
            tag = subprocess.check_output(
                [
                    'git',
                    '--no-pager',
                    'describe',
                    '--abbrev=0',
                    '--tags',
                ],
                stderr=subprocess.DEVNULL,
            ).strip().decode()
        except subprocess.CalledProcessError as e:
            tag = 'v0.0.0'

        version = tag[1:]

        nocolor()
        confoverrides = {}
        confoverrides['project'] = name
        confoverrides['version'] = version
        confdir = os.path.join(project_dir, 'docs')
        srcdir = confdir
        builder = 'html'
        build = self.get_finalized_command('build')
        build_dir = os.path.join(os.path.abspath(build.build_base), 'sphinx')
        builder_target_dir = os.path.join(build_dir, builder)
        app = None

        try:
            with patch_docutils(confdir), docutils_namespace():
                app = Sphinx(
                    srcdir,
                    confdir,
                    builder_target_dir,
                    os.path.join(build_dir, 'doctrees'),
                    builder,
                    confoverrides,
                    sys.stdout,
                    freshenv=False,
                    warningiserror=self.wall,
                    verbosity=self.distribution.verbose - 1,
                    keep_going=False,
                )
                app.build(force_all=False)
                if app.statuscode:
                    sys.stderr.write(
                        'Sphinx builder {} failed.'.format(app.builder.name),
                    )
                    raise SystemExit(8)
        except Exception as e:
            handle_exception(app, self, e, sys.stderr)
            raise


if __name__ == '__main__':
    setup(
        use_scm_version=True,
        long_description=open('README.md').read(),
        long_description_content_type="text/markdown",
        cmdclass={
            'apidoc': SphinxApiDoc,
            'build_sphinx': SphinxDoc,
        },
    )
