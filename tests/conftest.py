import os
import tempfile
import warnings

os.environ.setdefault("ORAN_ASM_CACHE", os.path.join(tempfile.gettempdir(), "oran_asm_test_cache"))
warnings.filterwarnings("ignore", message=".*SYNTHETIC.*")
