import unittest
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent.parent


class EnvironmentDefinitionTests(unittest.TestCase):
    def test_unified_py310_environment_files_are_portable(self):
        env_file = ROOT / 'environment_py310.yaml'
        mamba_env_file = ROOT / 'environment_mamba.yaml'
        base_requirements = ROOT / 'requirements_py310.txt'
        mamba_requirements = ROOT / 'requirements_mamba.txt'

        self.assertTrue(env_file.exists(), 'environment_py310.yaml must exist')
        self.assertTrue(mamba_env_file.exists(), 'environment_mamba.yaml must exist')
        self.assertTrue(base_requirements.exists(), 'requirements_py310.txt must exist')
        self.assertTrue(mamba_requirements.exists(), 'requirements_mamba.txt must exist')

        env_text = env_file.read_text(encoding='utf-8')
        mamba_env_text = mamba_env_file.read_text(encoding='utf-8')
        base_text = base_requirements.read_text(encoding='utf-8')
        mamba_text = mamba_requirements.read_text(encoding='utf-8')

        self.assertIn('name: mmvr-py310', env_text)
        self.assertIn('- python=3.10', env_text)
        self.assertIn('- -r requirements_mamba.txt', env_text)
        self.assertIn('pytorch=2.5.1', env_text)
        self.assertIn('pytorch-cuda=11.8', env_text)
        self.assertIn('pytorch=2.5.1', mamba_env_text)
        self.assertIn('pytorch-cuda=11.8', mamba_env_text)

        self.assertNotIn('file:///', base_text)
        self.assertNotIn('file:///', mamba_text)
        self.assertNotIn('-r requirements.txt', mamba_text)
        self.assertIn('-r requirements_py310.txt', mamba_text)
        self.assertIn('tensorboardX', base_text)
        self.assertIn('timm==1.0.24', base_text)
        self.assertIn('causal_conv1d-1.6.0+cu11torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl', mamba_text)
        self.assertIn('mamba_ssm-2.3.0+cu11torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl', mamba_text)
        self.assertNotIn('torch==1.13.1', mamba_text)

        setup_script = ROOT / 'scripts' / 'setup_mmvr_py310.sh'
        self.assertTrue(setup_script.exists(), 'scripts/setup_mmvr_py310.sh must exist')
        script_text = setup_script.read_text(encoding='utf-8')
        self.assertIn('TORCH_VERSION="${TORCH_VERSION:-2.5.1}"', script_text)
        self.assertIn('TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"', script_text)
        self.assertIn('TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"', script_text)
        self.assertIn('TIMM_VERSION="${TIMM_VERSION:-1.0.24}"', script_text)
        self.assertIn('PYTORCH_INDEX_MODE="${PYTORCH_INDEX_MODE:-official-cu118}"', script_text)
        self.assertIn('conda create -n "$ENV_NAME" python=3.10 pip -y', script_text)
        self.assertIn('requirements_py310.txt', script_text)
        self.assertIn('mamba_ssm-', script_text)
        self.assertIn('CAUSAL_CONV1D_WHL_URL', script_text)
        self.assertIn('MAMBA_SSM_WHL_URL', script_text)
        self.assertIn('download_wheel()', script_text)
        self.assertIn('causal_conv1d-1.6.0+cu11torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl', script_text)
        self.assertIn('mamba_ssm-2.3.0+cu11torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl', script_text)
        self.assertIn('case "$PYTORCH_INDEX_MODE" in', script_text)
        self.assertIn('mirror)', script_text)
        self.assertIn('official-cu118)', script_text)
        self.assertIn('--index-url "$TORCH_WHEEL_INDEX"', script_text)
        self.assertIn('log_step()', script_text)
        self.assertIn('DEBUG="${DEBUG:-0}"', script_text)
        self.assertIn('python -m pip install -r "$REPO_ROOT/requirements_py310.txt"', script_text)
        self.assertIn('python -m pip install "$causal_wheel"', script_text)
        self.assertIn('python -m pip install "$mamba_wheel"', script_text)
        self.assertNotIn('torch==1.13.1+cu116', script_text)

    def test_download_wheel_returns_clean_path_without_log_prefix(self):
        setup_script = ROOT / 'scripts' / 'setup_mmvr_py310.sh'

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_bin = temp_path / 'bin'
            fake_bin.mkdir()
            fake_wget = fake_bin / 'wget'
            fake_wget.write_text(
                '#!/usr/bin/env bash\n'
                'set -euo pipefail\n'
                'printf wheel > "$2"\n',
                encoding='utf-8',
            )
            fake_wget.chmod(0o755)

            dest_dir = temp_path / 'wheels'
            url = 'https://example.com/test.whl'
            command = f'''set -euo pipefail
source <(sed '/^ensure_conda$/,$d' "{setup_script}")
PATH="{fake_bin}:$PATH"
result="$(download_wheel "{url}" "{dest_dir}")"
printf '%s' "$result"
'''

            completed = subprocess.run(
                ['bash', '-lc', command],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertEqual(str(dest_dir / 'test.whl'), completed.stdout)


if __name__ == '__main__':
    unittest.main()
