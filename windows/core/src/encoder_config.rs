//! Explicit encoding of resolved device configuration and GPU capabilities.
//!
//! Native construction reads every field from the API snapshot and never reapplies defaults.

use mtld3d_shared::{
    encoder_wire::{WireError, WireReader, WireWriter},
    mtl::{ColorSpacePolicy, DeviceCapsFlags, SoftwareCursorPolicy},
};

use crate::{
    config::{AdapterSpoof, CursorScale, Mtld3dConfig},
    encoder_value::WireValue,
    gpu_caps::GpuCaps,
};

#[cfg(test)]
mod tests;

impl WireValue for Mtld3dConfig {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        self.caps_all.write_wire(writer)?;
        self.main_thread_checker.write_wire(writer)?;
        self.fail_next_submit.write_wire(writer)?;
        self.expand_packed16.write_wire(writer)?;
        self.deny_float32_filtering.write_wire(writer)?;
        self.managed_memory.write_wire(writer)?;
        self.linear_align256.write_wire(writer)?;
        self.hdr_enable.write_wire(writer)?;
        self.scene_float.write_wire(writer)?;
        self.color_space.write_wire(writer)?;
        self.cursor_scale.write_wire(writer)?;
        self.cursor_software.write_wire(writer)?;
        self.shader_cache_enable.write_wire(writer)?;
        self.shader_async_compile.write_wire(writer)?;
        self.log_dir.write_wire(writer)?;
        self.bytecode_dump_dir.write_wire(writer)?;
        self.skip_shaders.write_wire(writer)?;
        self.present_gate_file.write_wire(writer)?;
        self.present_occluded.write_wire(writer)?;
        self.query_flush_immediate.write_wire(writer)?;
        self.query_event_immediate.write_wire(writer)?;
        self.depth_alias_same_size.write_wire(writer)?;
        self.buffer_ignore_lock_bounds.write_wire(writer)?;
        self.vbib_retention_cap_bytes.write_wire(writer)?;
        self.vram_budget_cap_bytes.write_wire(writer)?;
        self.pagebox_pool_cap_bytes.write_wire(writer)?;
        self.present_max_fps.write_wire(writer)?;
        self.present_max_frame_latency.write_wire(writer)?;
        self.render_scale_percent.write_wire(writer)?;
        self.render_lod_bias.write_wire(writer)?;
        self.adapter_spoof.write_wire(writer)?;
        self.df_formats.write_wire(writer)?;
        self.ppl.write_wire(writer)?;
        self.hemi.write_wire(writer)?;
        self.lights.write_wire(writer)?;
        self.fog.write_wire(writer)?;
        self.water.write_wire(writer)?;
        self.post.write_wire(writer)?;
        Ok(())
    }

    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        Ok(Self {
            caps_all: <bool>::read_wire(reader)?,
            main_thread_checker: <bool>::read_wire(reader)?,
            fail_next_submit: <bool>::read_wire(reader)?,
            expand_packed16: <bool>::read_wire(reader)?,
            deny_float32_filtering: <bool>::read_wire(reader)?,
            managed_memory: <bool>::read_wire(reader)?,
            linear_align256: <bool>::read_wire(reader)?,
            hdr_enable: <bool>::read_wire(reader)?,
            scene_float: <bool>::read_wire(reader)?,
            color_space: <ColorSpacePolicy>::read_wire(reader)?,
            cursor_scale: <CursorScale>::read_wire(reader)?,
            cursor_software: <SoftwareCursorPolicy>::read_wire(reader)?,
            shader_cache_enable: <bool>::read_wire(reader)?,
            shader_async_compile: <bool>::read_wire(reader)?,
            log_dir: <String>::read_wire(reader)?,
            bytecode_dump_dir: <String>::read_wire(reader)?,
            skip_shaders: <Vec<u64>>::read_wire(reader)?,
            present_gate_file: <String>::read_wire(reader)?,
            present_occluded: <bool>::read_wire(reader)?,
            query_flush_immediate: <bool>::read_wire(reader)?,
            query_event_immediate: <bool>::read_wire(reader)?,
            depth_alias_same_size: <bool>::read_wire(reader)?,
            buffer_ignore_lock_bounds: <bool>::read_wire(reader)?,
            vbib_retention_cap_bytes: <u64>::read_wire(reader)?,
            vram_budget_cap_bytes: <u64>::read_wire(reader)?,
            pagebox_pool_cap_bytes: <u64>::read_wire(reader)?,
            present_max_fps: <u32>::read_wire(reader)?,
            present_max_frame_latency: <u32>::read_wire(reader)?,
            render_scale_percent: <u32>::read_wire(reader)?,
            render_lod_bias: <bool>::read_wire(reader)?,
            adapter_spoof: <AdapterSpoof>::read_wire(reader)?,
            df_formats: <bool>::read_wire(reader)?,
            ppl: <crate::ppl::PplConfig>::read_wire(reader)?,
            hemi: <Option<crate::ppl::HemiAmbient>>::read_wire(reader)?,
            lights: <crate::light_controls::LightControls>::read_wire(reader)?,
            fog: <crate::fog::FogConfig>::read_wire(reader)?,
            water: <crate::water::WaterConfig>::read_wire(reader)?,
            post: <crate::post::PostConfig>::read_wire(reader)?,
        })
    }
}

impl WireValue for crate::fog::FogConfig {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        use crate::fog::FogCurve;
        self.figures.write_wire(writer)?;
        writer.u8(match self.curve {
            FogCurve::Linear => 0,
            FogCurve::Exp => 1,
            FogCurve::Exp2 => 2,
        })?;
        writer.u8(self.desaturate_pct)
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        use crate::fog::FogCurve;
        Ok(Self {
            figures: <bool>::read_wire(reader)?,
            curve: match reader.u8()? {
                0 => FogCurve::Linear,
                1 => FogCurve::Exp,
                2 => FogCurve::Exp2,
                _ => return Err(WireError::InvalidValue),
            },
            desaturate_pct: reader.u8()?,
        })
    }
}

impl WireValue for crate::water::WaterConfig {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        self.fresnel.write_wire(writer)?;
        writer.u16(self.glint_pct)
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        Ok(Self {
            fresnel: <bool>::read_wire(reader)?,
            glint_pct: reader.u16()?,
        })
    }
}

impl WireValue for crate::post::PostConfig {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        self.debug.write_wire(writer)?;
        writer.u16(self.ao_pct)?;
        writer.u16(self.god_rays_pct)?;
        writer.u16(self.bloom_pct)?;
        writer.u16(self.bloom_threshold_pct)
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        Ok(Self {
            debug: <bool>::read_wire(reader)?,
            ao_pct: reader.u16()?,
            god_rays_pct: reader.u16()?,
            bloom_pct: reader.u16()?,
            bloom_threshold_pct: reader.u16()?,
        })
    }
}

impl WireValue for crate::ppl::PplConfig {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        use crate::ppl::{PplFalloff, PplMode, PplModel};
        writer.u8(match self.mode {
            PplMode::Off => 0,
            PplMode::Points => 1,
            PplMode::All => 2,
        })?;
        self.mixed.write_wire(writer)?;
        writer.u8(u8::from(matches!(self.model, PplModel::HalfLambert)))?;
        writer.u8(u8::from(matches!(self.falloff, PplFalloff::Smooth)))?;
        self.debug_term.write_wire(writer)?;
        writer.u8(self.shoulder_pct)?;
        writer.u8(u8::from(self.figures))?;
        self.specular.write_wire(writer)?;
        writer.u16(self.specular_scale_pct)?;
        self.specular_power.write_wire(writer)?;
        self.specular_default_pct.write_wire(writer)
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        use crate::ppl::{PplFalloff, PplMode, PplModel};
        let mode = match reader.u8()? {
            0 => PplMode::Off,
            1 => PplMode::Points,
            2 => PplMode::All,
            _ => return Err(WireError::InvalidValue),
        };
        let mixed = <bool>::read_wire(reader)?;
        let model = match reader.u8()? {
            0 => PplModel::D3d9,
            1 => PplModel::HalfLambert,
            _ => return Err(WireError::InvalidValue),
        };
        let falloff = match reader.u8()? {
            0 => PplFalloff::D3d9,
            1 => PplFalloff::Smooth,
            _ => return Err(WireError::InvalidValue),
        };
        Ok(Self {
            mode,
            mixed,
            model,
            falloff,
            debug_term: <bool>::read_wire(reader)?,
            shoulder_pct: reader.u8()?,
            figures: reader.u8()? != 0,
            specular: <bool>::read_wire(reader)?,
            specular_scale_pct: reader.u16()?,
            specular_power: <Option<u8>>::read_wire(reader)?,
            specular_default_pct: <Option<[u16; 3]>>::read_wire(reader)?,
        })
    }
}

impl WireValue for crate::light_controls::Scaled {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        writer.u16(self.scale_pct)?;
        writer.u16(self.tint_pct[0])?;
        writer.u16(self.tint_pct[1])?;
        writer.u16(self.tint_pct[2])
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        Ok(Self {
            scale_pct: reader.u16()?,
            tint_pct: [reader.u16()?, reader.u16()?, reader.u16()?],
        })
    }
}

impl WireValue for crate::light_controls::LightControls {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        self.sun.write_wire(writer)?;
        self.ambient.write_wire(writer)?;
        self.night_floor_pct.write_wire(writer)?;
        self.torch.write_wire(writer)?;
        self.placed_scale_pct.write_wire(writer)?;
        self.placed_range_pct.write_wire(writer)?;
        self.figure_scale_pct.write_wire(writer)
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        Ok(Self {
            sun: <crate::light_controls::Scaled>::read_wire(reader)?,
            ambient: <crate::light_controls::Scaled>::read_wire(reader)?,
            night_floor_pct: <u16>::read_wire(reader)?,
            torch: <crate::light_controls::Scaled>::read_wire(reader)?,
            placed_scale_pct: <u16>::read_wire(reader)?,
            placed_range_pct: <u16>::read_wire(reader)?,
            figure_scale_pct: <u16>::read_wire(reader)?,
        })
    }
}

impl WireValue for crate::ppl::HemiAmbient {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        writer.u16(self.sky_pct)?;
        writer.u16(self.ground_pct)
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        let sky_pct = reader.u16()?;
        let ground_pct = reader.u16()?;
        Ok(Self {
            sky_pct,
            ground_pct,
        })
    }
}

impl WireValue for AdapterSpoof {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        writer.u8(match self {
            Self::None => 0,
            Self::Nvidia => 1,
            Self::Amd => 2,
        })
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        match reader.u8()? {
            0 => Ok(Self::None),
            1 => Ok(Self::Nvidia),
            2 => Ok(Self::Amd),
            _ => Err(WireError::InvalidValue),
        }
    }
}

impl WireValue for ColorSpacePolicy {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        writer.u8(match self {
            Self::Passthrough => 0,
            Self::Accurate => 1,
        })
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        match reader.u8()? {
            0 => Ok(Self::Passthrough),
            1 => Ok(Self::Accurate),
            _ => Err(WireError::InvalidValue),
        }
    }
}

impl WireValue for SoftwareCursorPolicy {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        writer.u8(match self {
            Self::Auto => 0,
            Self::On => 1,
            Self::Off => 2,
        })
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        match reader.u8()? {
            0 => Ok(Self::Auto),
            1 => Ok(Self::On),
            2 => Ok(Self::Off),
            _ => Err(WireError::InvalidValue),
        }
    }
}

impl WireValue for CursorScale {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        match self {
            Self::Auto => writer.u8(0),
            Self::Fixed(value) => {
                writer.u8(1)?;
                writer.u32(*value)
            }
        }
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        match reader.u8()? {
            0 => Ok(Self::Auto),
            1 => Ok(Self::Fixed(reader.u32()?)),
            _ => Err(WireError::InvalidValue),
        }
    }
}

impl WireValue for GpuCaps {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        self.unified_memory.write_wire(writer)?;
        self.min_linear_texture_align.write_wire(writer)?;
        self.device_caps.bits().write_wire(writer)
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        Ok(Self {
            unified_memory: bool::read_wire(reader)?,
            min_linear_texture_align: u32::read_wire(reader)?,
            device_caps: DeviceCapsFlags::from_bits(u32::read_wire(reader)?)
                .ok_or(WireError::InvalidValue)?,
        })
    }
}
