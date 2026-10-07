"""Rendering-only industrial finish. The binary model and dynamics stay intact."""
import json
import numpy as np
from ..paths import CONFIG


# The refined assets use numerical material names; retain their authoring meaning.
REFINED_CLASSES = {
    0: 'metal', 2: 'metal', 4: 'paint', 5: 'glass', 7: 'rubber',
    8: 'emissive', 9: 'floor', 13: 'concrete', 20: 'emissive',
    25: 'wood', 26: 'wood', 28: 'metal', 29: 'metal', 30: 'rubber',
    31: 'rubber', 32: 'metal',
}


def material_class(name):
    if name.startswith('refined_mat_'):
        return REFINED_CLASSES.get(int(name.rsplit('_', 1)[1]), 'paint')
    if any(s in name for s in ('rubber', 'polymer', 'belt', 'cloth', 'trouser')):
        return 'rubber'
    if any(s in name for s in ('glass', 'optics')):
        return 'glass'
    if any(s in name for s in ('metal', 'silver', 'alloy', 'steel', 'stainless', 'galvanized', 'solder', 'copper', 'gold', 'edge', 'anodized')):
        return 'metal'
    if any(s in name for s in ('concrete', 'paver')):
        return 'concrete'
    if any(s in name for s in ('floor', 'aisle', 'asphalt')):
        return 'floor'
    if any(s in name for s in ('wood', 'cardboard')):
        return 'wood'
    if any(s in name for s in ('emissive', 'led', 'display')):
        return 'emissive'
    return 'paint'


class IndustrialAppearance:
    def __init__(self, model):
        self.config = json.loads((CONFIG / 'appearance_industrial.json').read_text())
        self.shadow_lights = []
        model.vis.quality.shadowsize = self.config['shadow_size']
        model.vis.quality.offsamples = 4
        model.vis.map.shadowclip = self.config['viewer_shadow_radius_m'] / model.stat.extent
        for key, value in self.config['headlight'].items():
            getattr(model.vis.headlight, key)[:] = value
        for i in range(model.nlight):
            values = self.config['lights'].get(model.light(i).name)
            if not values:
                continue
            for key, value in values.items():
                getattr(model, 'light_' + key)[i] = value
            model.light_dir[i] /= np.linalg.norm(model.light_dir[i])
            if values['castshadow']:
                self.shadow_lights.append(i)
        for i in range(model.nmat):
            name = model.mat(i).name
            values = self.config['material_classes'][material_class(name)]
            for key, value in values.items():
                getattr(model, 'mat_' + key)[i] = value
            color = self.config['color_overrides'].get(name)
            if color is not None:
                model.mat_rgba[i, :3] = color
                # Geom RGBA overrides material RGBA in the classic renderer.
                model.geom_rgba[model.geom_matid == i, :3] = color

    def prepare(self, model, lookat, distance, renderer=None):
        """Focus directional shadows without changing any geometry/camera clipping.

        Classic MuJoCo stores shadowClip on its GL context, so a reused renderer
        must update that value too. This adapter targets our pinned 3.12.1 runtime.
        """
        radius = max(self.config['shadow_min_radius_m'], float(distance) * 1.15)
        model.vis.map.shadowclip = radius / model.stat.extent
        for i in self.shadow_lights:
            model.light_pos[i] = np.asarray(lookat) - model.light_dir[i] * 500.
        if renderer is not None:
            renderer._mjr_context.shadowClip = radius
