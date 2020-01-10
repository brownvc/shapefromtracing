#pragma once

#include "redner.h"
#include "vector.h"
#include "ptr.h"
#include "atomic.h"
#include "assert.h"

template <int N>
struct Texture {
    Texture() {}
    Texture(ptr<float> texels,
            int width,
            int height,
            int channels, // ignored if N=-1
            int num_levels,
			int mesh_colors_resolution,
            ptr<float> uv_scale)
        : texels(texels.get()),
          width(width), height(height), channels(channels),
          num_levels(num_levels),
		  mesh_colors_resolution(mesh_colors_resolution),
          uv_scale(uv_scale.get()) {}

    float *texels;
    int width;
    int height;
    int channels;
    int num_levels;
	int mesh_colors_resolution;
    float *uv_scale;
};

using TextureN = Texture<-1>;
using Texture3 = Texture<3>;
using Texture1 = Texture<1>;

template <int N>
DEVICE
inline void trilinear_interp(const Texture<N> &tex,
                             int xfi, int yfi,
                             int xci, int yci,
                             Real u, Real v,
                             Real level,
                             Real *output) {
    // If channels == N, hopefully the constant would propagate and simplify the code.
    auto channels = N == -1 ? tex.channels : N;
    if (level <= 0 || level >= tex.num_levels - 1) {
        auto li = level <= 0 ? 0 : tex.num_levels - 1;
        auto texels = tex.texels + li * tex.width * tex.height * channels;
        for (int i = 0; i < channels; i++) {
            auto value_ff = texels[channels * (yfi * tex.width + xfi) + i];
            auto value_cf = texels[channels * (yfi * tex.width + xci) + i];
            auto value_fc = texels[channels * (yci * tex.width + xfi) + i];
            auto value_cc = texels[channels * (yci * tex.width + xci) + i];
            output[i] = value_ff * (1.f - u) * (1.f - v) +
                        value_fc * (1.f - u) *        v  +
                        value_cf *        u  * (1.f - v) +
                        value_cc *        u  *        v;
        }        
    } else {
        auto li = (int)floor(level);
        auto ld = level - li;
        auto texels0 = tex.texels + li * tex.width * tex.height * channels;
        auto texels1 = tex.texels + (li + 1) * tex.width * tex.height * channels;
        for (int i = 0; i < channels; i++) {
            auto value_ff0 = texels0[channels * (yfi * tex.width + xfi) + i];
            auto value_cf0 = texels0[channels * (yfi * tex.width + xci) + i];
            auto value_fc0 = texels0[channels * (yci * tex.width + xfi) + i];
            auto value_cc0 = texels0[channels * (yci * tex.width + xci) + i];
            auto value_ff1 = texels1[channels * (yfi * tex.width + xfi) + i];
            auto value_cf1 = texels1[channels * (yfi * tex.width + xci) + i];
            auto value_fc1 = texels1[channels * (yci * tex.width + xfi) + i];
            auto value_cc1 = texels1[channels * (yci * tex.width + xci) + i];
            auto v0 = value_ff0 * (1.f - u) * (1.f - v) +
                      value_fc0 * (1.f - u) *        v  +
                      value_cf0 *        u  * (1.f - v) +
                      value_cc0 *        u  *        v;
            auto v1 = value_ff1 * (1.f - u) * (1.f - v) +
                      value_fc1 * (1.f - u) *        v  +
                      value_cf1 *        u  * (1.f - v) +
                      value_cc1 *        u  *        v;
            output[i] = v0 * (1 - ld) + v1 * ld;
        }
    }
}

template <int N>
DEVICE
inline void d_trilinear_interp(const Texture<N> &tex,
                               int xfi, int yfi,
                               int xci, int yci,
                               Real u, Real v,
                               Real level,
                               const Real *d_output,
                               Texture<N> &d_tex,
                               Real &d_u, Real &d_v,
                               Real &d_level) {
    // If channels == N, hopefully the constant would propagate and simplify the code.
    auto channels = N == -1 ? tex.channels : N;
    if (level <= 0 || level >= tex.num_levels - 1) {
        auto li = level <= 0 ? 0 : tex.num_levels - 1;
        auto texels = tex.texels + li * tex.width * tex.height * channels;
        auto d_texels = d_tex.texels + li * tex.width * tex.height * channels;
        for (int i = 0; i < channels; i++) {
            auto value_ff = texels[channels * (yfi * tex.width + xfi) + i];
            auto value_cf = texels[channels * (yfi * tex.width + xci) + i];
            auto value_fc = texels[channels * (yci * tex.width + xfi) + i];
            auto value_cc = texels[channels * (yci * tex.width + xci) + i];
            // output[i] = value_ff * (1.f - u) * (1.f - v) +
            //             value_fc * (1.f - u) *        v  +
            //             value_cf *        u  * (1.f - v) +
            //             value_cc *        u  *        v;
            // d_value_ff
            atomic_add(&d_texels[channels * (yfi * tex.width + xfi) + i],
                       d_output[i] * (1.f - u) * (1.f - v));
            // d_value_fc
            atomic_add(&d_texels[channels * (yfi * tex.width + xci) + i],
                       d_output[i] *        u  * (1.f - v));
            // d_value_cf
            atomic_add(&d_texels[channels * (yci * tex.width + xfi) + i],
                       d_output[i] * (1.f - u) *        v );
            // d_value_cc
            atomic_add(&d_texels[channels * (yci * tex.width + xci) + i],
                       d_output[i] *        u  *        v );
            d_u += sum(d_output[i] * (-value_ff * (1.f - v) +
                                       value_cf * (1.f - v) +
                                      -value_fc *        v  +
                                       value_cc *        v));
            d_v += sum(d_output[i] * (-value_ff * (1.f - u) +
                                      -value_cf *        u  +
                                       value_fc * (1.f - u) +
                                       value_cc *        u));
        }
    } else {
        auto li = (int)floor(level);
        auto ld = level - li;
        auto texels0 = tex.texels + li * tex.width * tex.height * channels;
        auto texels1 = tex.texels + (li + 1) * tex.width * tex.height * channels;
        auto d_texels0 = d_tex.texels + li * tex.width * tex.height * channels;
        auto d_texels1 = d_tex.texels + (li + 1) * tex.width * tex.height * channels;
        for (int i = 0; i < channels; i++) {
            auto value_ff0 = texels0[channels * (yfi * tex.width + xfi) + i];
            auto value_cf0 = texels0[channels * (yfi * tex.width + xci) + i];
            auto value_fc0 = texels0[channels * (yci * tex.width + xfi) + i];
            auto value_cc0 = texels0[channels * (yci * tex.width + xci) + i];
            auto value_ff1 = texels1[channels * (yfi * tex.width + xfi) + i];
            auto value_cf1 = texels1[channels * (yfi * tex.width + xci) + i];
            auto value_fc1 = texels1[channels * (yci * tex.width + xfi) + i];
            auto value_cc1 = texels1[channels * (yci * tex.width + xci) + i];
            auto v0 = value_ff0 * (1.f - u) * (1.f - v) +
                      value_fc0 * (1.f - u) *        v  +
                      value_cf0 *        u  * (1.f - v) +
                      value_cc0 *        u  *        v;
            auto v1 = value_ff1 * (1.f - u) * (1.f - v) +
                      value_fc1 * (1.f - u) *        v  +
                      value_cf1 *        u  * (1.f - v) +
                      value_cc1 *        u  *        v;
            // output[i] = v0 * (1 - ld) + v1 * ld;
            auto d_v0 = d_output[i] * (1 - ld);
            auto d_v1 = d_output[i] *      ld;
            d_level += d_output[i] * (v1 - v0);
            // d_value_ff0
            atomic_add(&d_texels0[channels * (yfi * tex.width + xfi) + i],
                       d_v0 * (1.f - u) * (1.f - v));
            // d_value_fc0
            atomic_add(&d_texels0[channels * (yfi * tex.width + xci) + i],
                       d_v0 *        u  * (1.f - v));
            // d_value_cf0
            atomic_add(&d_texels0[channels * (yci * tex.width + xfi) + i],
                       d_v0 * (1.f - u) *        v );
            // d_value_cc0
            atomic_add(&d_texels0[channels * (yci * tex.width + xci) + i],
                       d_v0 *        u  *        v );
            // d_value_ff1
            atomic_add(&d_texels1[channels * (yfi * tex.width + xfi) + i],
                       d_v1 * (1.f - u) * (1.f - v));
            // d_value_fc1
            atomic_add(&d_texels1[channels * (yfi * tex.width + xci) + i],
                       d_v1 *        u  * (1.f - v));
            // d_value_cf1
            atomic_add(&d_texels1[channels * (yci * tex.width + xfi) + i],
                       d_v1 * (1.f - u) *        v );
            // d_value_cc1
            atomic_add(&d_texels1[channels * (yci * tex.width + xci) + i],
                       d_v1 *        u  *        v );
            d_u += d_v0 * (-value_ff0 * (1.f - v) +
                            value_cf0 * (1.f - v) +
                           -value_fc0 *        v  +
                            value_cc0 *        v);
            d_u += d_v1 * (-value_ff1 * (1.f - v) +
                            value_cf1 * (1.f - v) +
                           -value_fc1 *        v  +
                            value_cc1 *        v);
            d_v += d_v0 * (-value_ff0 * (1.f - u) +
                           -value_cf0 *        u  +
                            value_fc0 * (1.f - u) +
                            value_cc0 *        u);
            d_v += d_v1 * (-value_ff1 * (1.f - u) +
                           -value_cf1 *        u  +
                            value_fc1 * (1.f - u) +
                            value_cc1 *        u);
        }
    }
}

const Real epsilon = 0.0001;

DEVICE
inline Real round_local(Real val) {
	return floor(0.5f + val);
}

DEVICE
inline int get_mesh_colors_index(const int tri_id, const int res, int i, int j) {
	return tri_id * round_local((res + 1) * (res + 2) / 2) + round_local(i * (2 * res - i + 3) / 2) + j;
}

template <int N>
DEVICE
inline void mesh_colors_interp(const Texture<N> &tex,
								const int tri_id,
								const Vector2 &uv,
								Real *output) {

	auto channels = N == -1 ? tex.channels : N;
	const int r = tex.mesh_colors_resolution;

	const int i = floor(r * uv.x);
	const int j = floor(r * uv.y);

	const Real w = 1.0 - uv.x - uv.y;

	const Real w_x = r * uv.x - i;
	const Real w_y = r * uv.y - j;
	const Real w_z = r * w - floor(r * w);

	auto mc_ind0 = get_mesh_colors_index(tri_id, r, i + 1, j);
	auto mc_ind1 = get_mesh_colors_index(tri_id, r, i, j + 1);
	auto mc_ind2 = get_mesh_colors_index(tri_id, r, i, j);

	if (w_x + w_y + w_z < epsilon) {
		for (int i = 0; i < channels; i++) {
			output[i] = tex.texels[channels * mc_ind2 + i];
            assert(isfinite(output[i]));
		}
        return;
	}

	if (w_x + w_y + w_z < 2.0 - epsilon) {
		for (int i = 0; i < channels; i++) {
			auto c_i1j = tex.texels[channels * mc_ind0 + i];
			auto c_ij1 = tex.texels[channels * mc_ind1 + i];
			auto c_ij = tex.texels[channels * mc_ind2 + i];

			output[i] = w_x * c_i1j + w_y * c_ij1 + w_z * c_ij;
            assert(isfinite(output[i]));
		}
	}
	else {
		mc_ind2 = get_mesh_colors_index(tri_id, r, i + 1, j + 1);

		for (int i = 0; i < channels; i++) {
			auto c_i1j = tex.texels[channels * mc_ind0 + i];
			auto c_ij1 = tex.texels[channels * mc_ind1 + i];
			auto c_i1j1 = tex.texels[channels * mc_ind2 + i];

			output[i] = (1.0 - w_x) * c_ij1 + (1.0 - w_y) * c_i1j + (1.0 - w_z) * c_i1j1;
            assert(isfinite(output[i]));
		}
	}
}

template <int N>
DEVICE
inline void d_mesh_colors_interp(const Texture<N> &tex,
								const int tri_id,
								const Vector2 &uv,
								const Real *d_output,
								Texture<N> &d_tex) {

	auto channels = N == -1 ? tex.channels : N;
	const int r = tex.mesh_colors_resolution;

	const int i = floor(r * uv.x);
	const int j = floor(r * uv.y);

	const Real w = 1.0 - uv.x - uv.y;

	const Real w_x = r * uv.x - i;
	const Real w_y = r * uv.y - j;
	const Real w_z = r * w - floor(r * w);

	if (w_x + w_y + w_z < epsilon) {
		// No derivatives here pardner     
		return;
	}

	auto mc_ind0 = get_mesh_colors_index(tri_id, r, i + 1, j);
	auto mc_ind1 = get_mesh_colors_index(tri_id, r, i, j + 1);
	auto mc_ind2 = (w_x + w_y + w_z < 2 - epsilon) ? 
		get_mesh_colors_index(tri_id, r, i, j) :
		get_mesh_colors_index(tri_id, r, i + 1, j + 1);

	for (int i = 0; i < channels; i++) {
        assert(isfinite(d_output[i] * w_x));
        assert(isfinite(d_output[i] * w_y));
        assert(isfinite(d_output[i] * w_z));

		atomic_add(&d_tex.texels[channels * mc_ind0 + i], d_output[i] * w_x);
		atomic_add(&d_tex.texels[channels * mc_ind1 + i], d_output[i] * w_y);
		atomic_add(&d_tex.texels[channels * mc_ind2 + i], d_output[i] * w_z);
	}
}

template <int N>
DEVICE
inline void get_texture_value_constant(const Texture<N> &tex,
                                       Real *output) {
    auto channels = N == -1 ? tex.channels : N;
    for (int i = 0; i < channels; i++) {
        output[i] = tex.texels[i];
    }
}

template <typename TextureType>
DEVICE
inline void get_texture_value(const TextureType &tex,
							  const int tri_id_,
                              const Vector2 &uv_,
							  const Vector2 &tri_uv_,
                              const Vector2 &du_dxy_,
                              const Vector2 &dv_dxy_,
                              Real *output) {
    if (tex.num_levels <= 0) {
        // Constant texture
        get_texture_value_constant(tex, output);
    } 
	else if (tex.mesh_colors_resolution > 0) {
		mesh_colors_interp(tex, tri_id_, tri_uv_, output);
	}
	else {
        // Trilinear interpolation
        auto uv_scale = Vector2f{tex.uv_scale[0], tex.uv_scale[1]};
        auto uv = uv_ * uv_scale;
        auto du_dxy = du_dxy_ * uv_scale[0];
        auto dv_dxy = dv_dxy_ * uv_scale[1];
        auto x = uv[0] * tex.width - 0.5f;
        auto y = uv[1] * tex.height - 0.5f;
        auto xf = (int)floor(x);
        auto yf = (int)floor(y);
        auto xc = xf + 1;
        auto yc = yf + 1;
        auto u = x - xf;
        auto v = y - yf;
        auto xfi = modulo(xf, tex.width);
        auto yfi = modulo(yf, tex.height);
        auto xci = modulo(xc, tex.width);
        auto yci = modulo(yc, tex.height);
        auto max_footprint = max(length(du_dxy) * tex.width, length(dv_dxy) * tex.height);
        auto level = log2(max(max_footprint, Real(1e-8f)));
        trilinear_interp(tex, xfi, yfi, xci, yci, u, v, level, output);
    }
}

template <int N>
DEVICE
inline void d_get_texture_value(const Texture<N> &tex,
								const int tri_id_,
                                const Vector2 &uv_,
								const Vector2 &tri_uv_,
                                const Vector2 &du_dxy_,
                                const Vector2 &dv_dxy_,
                                const Real *d_output,
                                Texture<N> &d_tex,
                                Vector2 &d_uv_,
                                Vector2 &d_du_dxy_,
                                Vector2 &d_dv_dxy_) {
    if (tex.num_levels <= 0) {
        // Constant texture
        // output[i] = tex.texels[i]
        auto channels = N == -1 ? tex.channels : N;
        for (int i = 0; i < channels; i++) {
            atomic_add(d_tex.texels[i], d_output[i]);
        }
	}
	else if (tex.mesh_colors_resolution > 0) {
		d_mesh_colors_interp(tex, tri_id_, tri_uv_, d_output, d_tex);
	} 
	else {
        // Trilinear interpolation
        auto uv_scale = Vector2f{tex.uv_scale[0], tex.uv_scale[1]};
        auto uv = uv_ * uv_scale;
        auto du_dxy = du_dxy_ * uv_scale[0];
        auto dv_dxy = dv_dxy_ * uv_scale[1];
        auto x = uv[0] * tex.width - 0.5f;
        auto y = uv[1] * tex.height - 0.5f;
        auto xf = (int)floor(x);
        auto yf = (int)floor(y);
        auto xc = xf + 1;
        auto yc = yf + 1;
        auto u = x - xf;
        auto v = y - yf;
        auto xfi = modulo(xf, tex.width);
        auto yfi = modulo(yf, tex.height);
        auto xci = modulo(xc, tex.width);
        auto yci = modulo(yc, tex.height);
        auto u_footprint = length(du_dxy) * tex.width;
        auto v_footprint = length(dv_dxy) * tex.height;
        bool is_u_max = true;
        auto max_footprint = u_footprint;
        if (v_footprint > u_footprint) {
            is_u_max = false;
            max_footprint = v_footprint;
        }
        auto level = log2(max(max_footprint, Real(1e-8f)));
        auto d_u = Real(0);
        auto d_v = Real(0);
        auto d_level = Real(0);
        d_trilinear_interp(tex, xfi, yfi, xci, yci, u, v, level,
                           d_output, d_tex, d_u, d_v, d_level);
        auto d_max_footprint = Real(0);
        // level = log2(max(max_footprint, Real(1e-8f)))
        if (max_footprint > Real(1e-8f)) {
            d_max_footprint += d_level / (max_footprint * log(Real(2)));
        }
        // max_footprint = max(length(du_dxy) * tex.width, length(dv_dxy) * tex.height)
        auto d_uv = Vector2{0, 0};
        auto d_du_dxy = Vector2{0, 0};
        auto d_dv_dxy = Vector2{0, 0};
        if (max_footprint > Real(1e-8f)) {
            if (is_u_max) {
                d_du_dxy += d_length(du_dxy, d_max_footprint) * tex.width;
            } else {
                d_dv_dxy += d_length(dv_dxy, d_max_footprint) * tex.height;
            }
        }

        // du = dx, dv = dy
        // x = uv[0] * tex.width - 0.5f
        // y = uv[1] * tex.height - 0.5f
        d_uv[0] += d_u * tex.width;
        d_uv[1] += d_v * tex.height;

        // uv = uv_ * uv_scale
        // du_dxy = du_dxy_ * uv_scale[0]
        // dv_dxy = dv_dxy_ * uv_scale[1]
        d_uv_ += d_uv * uv_scale;
        d_du_dxy_ += d_du_dxy * uv_scale[0];
        d_dv_dxy_ += d_dv_dxy * uv_scale[1];
        atomic_add(d_tex.uv_scale,
            d_uv * uv_ + Vector2{sum(d_du_dxy * du_dxy_), sum(d_dv_dxy * dv_dxy_)});
    }
}

