from .functions.roi_align_rotated import roi_align_rotated
from .modules.roi_align_rotated import RoIAlignRotated

from .functions.aroi_align_rotated import aroi_align_rotated
from .modules.aroi_align_rotated import ARoIAlignRotated

__all__ = ['roi_align_rotated', 'RoIAlignRotated',
            'aroi_align_rotated', 'ARoIAlignRotated']
