//! Per-type views of a selected asset.
//!
//! Each viewer lives in its own module and is reached only through [`Viewer`], so adding a type is a
//! new file plus an arm here rather than edits threaded through the browser.

use egui::{
    Align, Color32, Label, Layout, Rect, RichText, ScrollArea, Sense, TextureHandle, Vec2, vec2,
};

use super::{Bytes, Channels, MAX_TEXT_PREVIEW};
use crate::utils::export;

pub mod atch;
pub mod avfx;
pub mod chara;
pub mod cmp;
pub mod cutb;
pub mod dic;
pub mod fpeb;
pub mod waoe;
pub mod wtd;
pub mod eid;
pub mod eqdp;
pub mod eqp;
pub mod est;
pub mod evp;
pub mod exl;
pub mod font;
pub mod gmp;
pub mod grass;
pub mod hwc;
pub mod icons;
pub mod imc;
pub mod layer;
pub mod luab;
pub mod material;
pub mod mdl;
pub mod pap;
pub mod pbd;
pub mod pcb;
pub mod phyb;
pub mod placed;
pub mod png;
pub mod scd;
mod shader;
pub mod shcd;
pub mod shpk;
pub mod skeleton;
pub mod sklb;
pub mod skp;
pub mod spm;
pub mod stm;
pub mod tera;
pub mod texture;
pub mod tmb;
pub mod uld;
pub mod zone;

/// Space kept around whatever a grid cell holds.
const PADDING: f32 = 6.0;

/// Uniform cells in as many columns as fit, virtualised by row: only the rows on screen are laid
/// out, which is what keeps a font of twenty-eight thousand glyphs from asking for every sheet it
/// names at once. Columns are counted from the width inside the scroll area, so the grid never
/// overflows into space there is no bar to reach.
fn grid(
    ui: &mut egui::Ui,
    cell: Vec2,
    count: usize,
    mut draw: impl FnMut(&mut egui::Ui, usize, Rect),
) {
    ScrollArea::vertical()
        .auto_shrink([false, false])
        .show_viewport(ui, |ui, viewport| {
            let width = ui.available_width();
            let columns = (width / cell.x).floor().max(1.0) as usize;
            let rows = count.div_ceil(columns);
            let origin =
                ui.cursor().left_top() + vec2((width - columns as f32 * cell.x) / 2.0, 0.0);
            ui.set_height(rows as f32 * cell.y);

            let first = (viewport.min.y / cell.y).floor().max(0.0) as usize;
            let last = ((viewport.max.y / cell.y).ceil() as usize).min(rows);
            for index in (first * columns)..(last * columns).min(count) {
                let at = Rect::from_min_size(
                    origin
                        + vec2(
                            (index % columns) as f32 * cell.x,
                            (index / columns) as f32 * cell.y,
                        ),
                    cell,
                );
                draw(ui, index, at);
            }
        });
}

/// Stand-in for a sheet that never arrived, in the space its art would have taken.
fn missing(ui: &egui::Ui, rect: Rect) {
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        "⚠",
        egui::FontId::default(),
        Color32::LIGHT_RED,
    );
}

/// Width held for a fact's label. Fixed rather than natural, so the grid's un-tracked label
/// column can't push the value column - and the row with it - past whatever room the value's own
/// wrap was given.
const FACT_LABEL_WIDTH: f32 = 150.0;

/// A table of label and value, which is most of what a details panel is.
fn facts(ui: &mut egui::Ui, id: &str, rows: &[(&'static str, String)]) {
    // A zero-height cell centers the label a half-line off the value's own centering.
    let label_height = ui.text_style_height(&egui::TextStyle::Body);
    egui::Grid::new(id)
        .num_columns(2)
        .striped(true)
        .show(ui, |ui| {
            for (label, value) in rows {
                ui.allocate_ui_with_layout(
                    vec2(FACT_LABEL_WIDTH, label_height),
                    Layout::left_to_right(Align::Center),
                    |ui| ui.add(Label::new(RichText::new(*label).weak()).truncate()),
                );
                // Wrapped rather than run on: a grid's cell is as wide as it likes, and one long
                // row would take the panel and the view beside it with it.
                ui.add(Label::new(RichText::new(value).monospace()).wrap());
                ui.allocate_space(vec2(ui.available_width(), 0.0));
                ui.end_row();
            }
        });
}

/// The textures a viewer draws from, where the file itself never names them. Returns one if it was
/// followed.
fn textures<'a>(
    ui: &mut egui::Ui,
    paths: impl IntoIterator<Item = (Option<&'a str>, &'a str)>,
) -> Option<String> {
    ui.add_space(8.0);
    ui.separator();
    ui.label(RichText::new("Textures").weak());
    ui.add_space(4.0);

    let mut follow = None;
    egui::Grid::new("textures")
        .num_columns(2)
        .striped(true)
        .show(ui, |ui| {
            for (label, path) in paths {
                if let Some(label) = label {
                    ui.label(RichText::new(label).weak());
                }
                if link(ui, crate::utils::file_name(path), path) {
                    follow = Some(path.to_owned());
                }
                ui.allocate_space(vec2(ui.available_width(), 0.0));
                ui.end_row();
            }
        });
    follow
}

/// A section title in the main area: the details panel's weak styling at heading size.
fn section(ui: &mut egui::Ui, title: &str) {
    ui.label(RichText::new(title).text_style(egui::TextStyle::Heading));
    ui.add_space(4.0);
}

/// A group heading inside a section, for the tables that come in several kinds.
fn heading(ui: &mut egui::Ui, text: &str) {
    ui.add_space(4.0);
    ui.label(RichText::new(text).weak());
    ui.add_space(4.0);
}

/// The weak header row above a striped table.
fn headers(ui: &mut egui::Ui, names: &[&str]) {
    for name in names {
        ui.label(RichText::new(*name).weak().small());
    }
    ui.allocate_space(vec2(ui.available_width(), 0.0));
    ui.end_row();
}

/// One row of a monospace table, each cell padded to its column so the header above and every row
/// below hold the same columns.
fn line<'a>(columns: &[(&str, usize)], cells: impl IntoIterator<Item = &'a str>) -> String {
    columns
        .iter()
        .zip(cells)
        .map(|((_, width), cell)| format!("{cell:<width$}  "))
        .collect()
}

/// Names the table's scroll area, so the header can be drawn at the offset it was left at. It is
/// the same for every table, so two of them may not share one `Ui`.
const TABLE: &str = "table_rows";

/// Where [`table`] keeps its scroll offset. `ScrollArea` hashes the salt it was handed rather than
/// the string behind it, so asking for the same id means salting it the same way first.
fn table_id(ui: &egui::Ui) -> egui::Id {
    ui.make_persistent_id(egui::IdSalt::new(TABLE))
}

/// A monospace table, virtualised by row for the formats whose row count is whatever the file
/// holds. The rows scroll both ways, since a wide one runs past a narrow window, and the columns
/// line up because the header and the rows are padded alike by [`line`].
///
/// The header stays above the scroll area rather than moving with the rows, so it is painted at
/// the rows' own horizontal offset rather than laid out beside them.
fn table(
    ui: &mut egui::Ui,
    columns: &[(&str, usize)],
    count: usize,
    mut row: impl FnMut(&mut egui::Ui, usize),
) {
    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);

    let offset =
        egui::scroll_area::State::load(ui.ctx(), table_id(ui)).map_or(0.0, |state| state.offset.x);
    let color = ui.visuals().weak_text_color();
    let header = ui.painter().layout_no_wrap(
        line(columns, columns.iter().map(|(name, _)| *name)),
        egui::TextStyle::Monospace.resolve(ui.style()),
        color,
    );
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), header.size().y),
        Sense::hover(),
    );
    ui.painter()
        .with_clip_rect(rect)
        .galley(rect.min - egui::vec2(offset, 0.0), header, color);

    // `show_rows` adds the spacing between rows itself, so what it wants is one row's own height.
    let height = ui.text_style_height(&egui::TextStyle::Monospace);
    ScrollArea::both()
        .id_salt(TABLE)
        .auto_shrink(false)
        .show_rows(ui, height, count, |ui, shown| {
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
            for index in shown {
                row(ui, index);
            }
        });
}

/// Half-float colors are linear and can exceed 1.0, so they are tone-mapped rather than clamped;
/// otherwise every bright row renders as flat white.
/// Side of the square a color is drawn in.
const CHIP: f32 = 10.0;

/// One color, drawn beside the numbers it came from.
fn chip(ui: &mut egui::Ui, color: Color32) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(CHIP), Sense::hover());
    ui.painter().rect_filled(rect, 2.0, color);
    response
}

fn swatch(color: [f32; 3]) -> Color32 {
    let map = |v: f32| ((v / (1.0 + v)).clamp(0.0, 1.0) * 255.0) as u8;
    Color32::from_rgb(map(color[0]), map(color[1]), map(color[2]))
}

/// A clickable id, with the hover and copy menu every crc-named value in the browser gets.
fn hashed(ui: &mut egui::Ui, kind: &str, name: &str, id: u32, dim: bool) {
    labelled(ui, kind, name, name, id, dim);
}

/// The same, drawn under a shorter label. Hovering still gives the whole name and its hash, so
/// nothing is lost by not spelling out a key's own name in every one of its values.
fn labelled(ui: &mut egui::Ui, kind: &str, name: &str, shown: &str, id: u32, dim: bool) {
    let text = RichText::new(shown).monospace();
    let response = ui.add(
        egui::Label::new(match dim {
            true => text.weak(),
            false => text,
        })
        .sense(Sense::click()),
    );
    crate::assets::crc_context(&response, kind, name, id);
}

/// A path rendered as a link: hyperlink color, pointer cursor, and the same hover and right-click
/// menu every other path in the browser gets. Returns whether it was followed.
pub(crate) fn link(ui: &mut egui::Ui, text: &str, path: &str) -> bool {
    let response = ui
        .add(
            egui::Label::new(
                RichText::new(text)
                    .monospace()
                    .color(ui.visuals().hyperlink_color),
            )
            .wrap()
            .sense(Sense::click()),
        )
        .on_hover_cursor(egui::CursorIcon::PointingHand);
    crate::assets::path_context(&response, path, None);
    response.clicked()
}

pub enum Preview {
    Text(String),
    Image {
        texture: TextureHandle,
        size: [usize; 2],
        /// Slice count for a volume texture; one for everything else.
        depth: u16,
        /// Channels the source format carries, which is what the channel toggles should offer.
        components: u8,
        /// Label/value pairs describing the source file.
        facts: Vec<(&'static str, String)>,
        mips: Vec<Mip>,
    },
    /// A parsed material, rendered as its own layout rather than a flat table.
    Material(Box<material::Rendered>),
    /// A model, drawn.
    Model(Box<mdl::Rendered>),
    /// A parsed UI layout.
    Uld(Box<uld::Rendered>),
    /// A parsed font.
    Font(Box<font::Rendered>),
    /// The parsed icon sheet.
    Icons(Box<icons::Rendered>),
    /// A read Lua chunk.
    Luab(Box<luab::Rendered>),
    /// A parsed shader package.
    Shpk(Box<shpk::Rendered>),
    /// A parsed shader.
    Shcd(Box<shcd::Rendered>),
    /// A parsed image change file.
    Imc(Box<imc::Rendered>),
    /// A parsed attach point file.
    Atch(Box<atch::Rendered>),
    /// A parsed visual effect.
    Avfx(Box<avfx::Rendered>),
    /// A parsed bind point file.
    Eid(Box<eid::Rendered>),
    /// A parsed skeleton template file.
    Est(Box<est::Rendered>),
    /// A parsed sheet list.
    Exl(Box<exl::Rendered>),
    /// A parsed skeleton parameter file.
    Skp(Box<skp::Rendered>),
    /// A parsed terrain file.
    Tera(Box<tera::Rendered>),
    /// A parsed staining template file.
    Stm(Box<stm::Rendered>),
    /// A parsed timeline.
    Tmb(Box<tmb::Rendered>),
    /// A parsed animation pack.
    Pap(Box<pap::Rendered>),
    /// A parsed physics file.
    Phyb(Box<phyb::Rendered>),
    /// A parsed skeleton.
    Sklb(Box<sklb::Rendered>),
    /// A parsed set of deformer parameters.
    Eqdp(Box<eqdp::Rendered>),
    /// A parsed set of equipment parameters.
    Eqp(Box<eqp::Rendered>),
    /// A parsed set of gimmick parameters.
    Gmp(Box<gmp::Rendered>),
    /// A parsed set of equipment VFX parameters.
    Evp(Box<evp::Rendered>),
    /// A parsed layer group, from either of the two files that hold one.
    Layers(Box<layer::Rendered>),
    /// A parsed annotation of what a zone's layers placed.
    Zone(Box<zone::instanced::Rendered>),
    /// A parsed environment set.
    Environments(Box<zone::envs::Rendered>),
    /// A parsed ambient light file.
    Ambient(Box<zone::amb::Rendered>),
    /// A parsed shader parameter map.
    Spm(Box<spm::Rendered>),
    /// A parsed pre-bone deformer.
    Pbd(Box<pbd::Rendered>),
    /// A parsed collision file.
    Pcb(Box<pcb::Rendered>),
    /// A parsed character make parameter file.
    Cmp(Box<cmp::Rendered>),
    /// A parsed index of a zone's grass grids.
    GrassZone(Box<grass::Zone>),
    /// A parsed grass grid.
    GrassGrid(Box<grass::Grid>),
    /// A parsed word dictionary.
    Dic(Box<dic::Rendered>),
    Fpeb(Box<fpeb::Rendered>),
    Waoe(Box<waoe::Rendered>),
    Wtd(Box<wtd::Rendered>),
    /// A parsed cutscene.
    Cutb(Box<cutb::Rendered>),
    /// A parsed sound container.
    Scd(Box<scd::Rendered>),
    /// Nothing to render; an empty message means the type simply has no viewer.
    Failed(String),
}

impl Preview {
    pub fn decode(
        ctx: &egui::Context,
        path: &str,
        bytes: &[u8],
        viewer: Viewer,
        mip: u8,
        channels: Channels,
    ) -> Self {
        let result = match viewer {
            Viewer::Text => Ok(Self::Text(
                String::from_utf8_lossy(&bytes[..bytes.len().min(MAX_TEXT_PREVIEW)]).into_owned(),
            )),
            Viewer::Image => png::decode(ctx, path, bytes, channels),
            Viewer::Texture => texture::decode(ctx, path, bytes, mip, channels),
            Viewer::Material => material::decode(path, bytes),
            Viewer::Model => mdl::decode(path, bytes),
            Viewer::Uld => uld::decode(path, bytes),
            Viewer::Font => font::decode(path, bytes),
            Viewer::Icons => icons::decode(path, bytes),
            Viewer::Luab => luab::decode(path, bytes),
            Viewer::Shpk => shpk::decode(path, bytes),
            Viewer::Shcd => shcd::decode(path, bytes),
            Viewer::Imc => imc::decode(path, bytes),
            Viewer::Atch => atch::decode(path, bytes),
            Viewer::Avfx => avfx::decode(path, bytes),
            Viewer::Eid => eid::decode(path, bytes),
            Viewer::Est => est::decode(path, bytes),
            Viewer::Exl => exl::decode(path, bytes),
            Viewer::Skp => skp::decode(path, bytes),
            Viewer::Tera => tera::decode(path, bytes),
            Viewer::Stm => stm::decode(path, bytes),
            Viewer::Tmb => tmb::decode(path, bytes),
            Viewer::Pap => pap::decode(path, bytes),
            Viewer::Phyb => phyb::decode(path, bytes),
            Viewer::Sklb => sklb::decode(path, bytes),
            Viewer::Eqdp => eqdp::decode(path, bytes),
            Viewer::Eqp => eqp::decode(path, bytes),
            Viewer::Gmp => gmp::decode(path, bytes),
            Viewer::Evp => evp::decode(path, bytes),
            Viewer::Hwc => hwc::decode(ctx, path, bytes, channels),
            Viewer::Lgb => layer::lgb::decode(path, bytes),
            Viewer::Sgb => layer::sgb::decode(path, bytes),
            Viewer::Lvb => layer::lvb::decode(path, bytes),
            Viewer::Svb => zone::instanced::sky_visibility(path, bytes),
            Viewer::Lcb => zone::instanced::clip_boxes(path, bytes),
            Viewer::Uwb => zone::instanced::underwater(path, bytes),
            Viewer::Envb => zone::envs::environment(path, bytes),
            Viewer::Obsb => zone::envs::object_behavior(path, bytes),
            Viewer::Essb => zone::envs::sound(path, bytes),
            Viewer::Amb => zone::amb::decode(path, bytes),
            Viewer::Spm => spm::decode(path, bytes),
            Viewer::Pbd => pbd::decode(path, bytes),
            Viewer::Pcb => pcb::decode(path, bytes),
            Viewer::Cmp => cmp::decode(path, bytes),
            Viewer::Gzd => grass::zone(path, bytes),
            Viewer::Ggd => grass::grid(path, bytes),
            Viewer::Dic => dic::decode(path, bytes),
            Viewer::Fpeb => fpeb::decode(path, bytes),
            Viewer::Waoe => waoe::decode(path, bytes),
            Viewer::Wtd => wtd::decode(path, bytes),
            Viewer::Cutb => cutb::decode(path, bytes),
            Viewer::Scd => scd::decode(path, bytes),
            Viewer::Raw => return Self::Failed(String::new()),
        };
        result.unwrap_or_else(|e| Self::Failed(e.to_string()))
    }

    /// Draws the preview. Returns a path when the user follows a link out of it, such as a
    /// material's texture.
    ///
    /// `bytes` is the file the preview was decoded from, still owned by the browser.
    pub fn ui(
        &self,
        ui: &mut egui::Ui,
        bytes: &[u8],
        slice: u16,
        deps: &mut crate::assets::deps::Deps,
        backend: &crate::backend::Backend,
    ) -> Option<String> {
        let mut follow = None;
        match self {
            Self::Material(material) => {
                follow = material::ui(ui, material, deps, backend);
            }
            Self::Model(model) => mdl::ui(ui, model, backend),
            Self::Uld(layout) => {
                follow = uld::ui(ui, layout, deps, backend);
            }
            Self::Font(font) => font::ui(ui, font, deps, backend),
            Self::Icons(icons) => icons::ui(ui, icons, deps, backend),
            Self::Luab(chunk) => luab::ui(ui, chunk),
            Self::Shpk(package) => shpk::ui(ui, package, bytes),
            Self::Shcd(code) => shcd::ui(ui, code, bytes),
            Self::Imc(change) => imc::ui(ui, change),
            Self::Atch(points) => atch::ui(ui, points),
            Self::Avfx(effect) => follow = avfx::ui(ui, effect, backend),
            Self::Eid(points) => follow = eid::ui(ui, points),
            Self::Est(templates) => follow = est::ui(ui, templates),
            Self::Exl(list) => follow = exl::ui(ui, list),
            Self::Skp(parameters) => follow = skp::ui(ui, parameters),
            Self::Tera(terrain) => follow = tera::ui(ui, terrain),
            Self::Tmb(timeline) => follow = tmb::ui(ui, timeline),
            Self::Pap(pack) => follow = pap::ui(ui, pack, backend),
            Self::Phyb(physics) => phyb::ui(ui, physics),
            Self::Sklb(skeleton) => sklb::ui(ui, skeleton),
            Self::Eqdp(parameters) => eqdp::ui(ui, parameters),
            Self::Eqp(parameters) => eqp::ui(ui, parameters),
            Self::Gmp(parameters) => gmp::ui(ui, parameters),
            Self::Evp(parameters) => evp::ui(ui, parameters),
            Self::Layers(layers) => follow = layer::ui(ui, layers, deps, backend),
            Self::Zone(annotations) => follow = zone::instanced::ui(ui, annotations),
            Self::Environments(set) => follow = zone::envs::ui(ui, set, deps, backend),
            Self::Ambient(light) => follow = zone::amb::ui(ui, light),
            Self::Spm(parameters) => spm::ui(ui, parameters),
            Self::Pbd(deformers) => pbd::ui(ui, deformers),
            Self::Pcb(collision) => follow = pcb::ui(ui, collision, backend),
            Self::Cmp(parameters) => cmp::ui(ui, parameters, deps, backend),
            Self::GrassZone(zone) => follow = grass::zone_ui(ui, zone, deps, backend),
            Self::GrassGrid(grid) => grass::grid_ui(ui, grid),
            Self::Dic(dictionary) => dic::ui(ui, dictionary),
            Self::Fpeb(held) => fpeb::ui(ui, held),
            Self::Waoe(held) => waoe::ui(ui, held),
            Self::Wtd(held) => wtd::ui(ui, held),
            Self::Cutb(cutscene) => follow = cutb::ui(ui, cutscene, backend),
            Self::Stm(templates) => stm::ui(ui, templates, deps, backend),
            Self::Scd(container) => scd::ui(ui, container),
            Self::Failed(e) if e.is_empty() => {
                ui.centered_and_justified(|ui| {
                    ui.label(RichText::new("No viewer for this file type. Use Raw bytes.").weak());
                });
            }
            Self::Failed(e) => {
                ui.centered_and_justified(|ui| {
                    ui.colored_label(Color32::RED, format!("Could not render this file: {e}"));
                });
            }
            Self::Text(text) => {
                ScrollArea::both().auto_shrink(false).show(ui, |ui| {
                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                    ui.add(Label::new(RichText::new(text).monospace()).selectable(true));
                });
            }
            Self::Image {
                texture,
                size,
                depth,
                ..
            } => {
                // The whole volume is resident as a grid of slices, so changing slice is only a
                // different uv rect into the cell it landed in.
                let (columns, rows) = crate::utils::tex_loader::grid_layout((*depth).max(1));
                let (column, row) = (slice % columns, slice / columns);
                let (columns, rows) = (f32::from(columns), f32::from(rows));
                let (left, top) = (f32::from(column) / columns, f32::from(row) / rows);
                let uv = egui::Rect::from_min_max(
                    egui::pos2(left, top),
                    egui::pos2(left + 1.0 / columns, top + 1.0 / rows),
                );
                // `uv` only changes what is sampled; the widget still sizes itself from the whole
                // texture unless the source size is stated as one slice.
                let slice_size = egui::vec2(
                    (size[0] as f32 / columns).max(1.0),
                    (size[1] as f32 / rows).max(1.0),
                );
                ScrollArea::both().auto_shrink(false).show(ui, |ui| {
                    let align = if slice_size.x < ui.available_width() {
                        Align::Center
                    } else {
                        Align::Min
                    };
                    ui.with_layout(Layout::top_down(align), |ui| {
                        ui.add(
                            egui::Image::new(egui::load::SizedTexture::new(
                                texture.id(),
                                slice_size,
                            ))
                            .uv(uv)
                            .fit_to_original_size(1.0),
                        );
                    });
                });
            }
        }
        follow
    }

    /// Export choices beyond the raw file, which the browser always offers on its own. `viewer` is
    /// what tells apart the several formats that decode into the same [`Self::Image`] shape.
    #[allow(clippy::too_many_arguments)]
    pub fn export_choices<'a>(
        &'a self,
        viewer: Viewer,
        path: &str,
        bytes: &'a [u8],
        mip: u8,
        ctx: &egui::Context,
    ) -> Vec<export::Choice<'a>> {
        match self {
            Self::Luab(chunk) => luab::export_choices(chunk),
            Self::Shpk(package) => shpk::export_choices(package, bytes, ctx),
            Self::Shcd(code) => shcd::export_choices(code, bytes),
            Self::Image { .. } if viewer == Viewer::Hwc => hwc::export_choices(bytes),
            Self::Image { .. } if viewer == Viewer::Texture => {
                texture_export_choices(bytes, path, mip)
            }
            _ => Vec::new(),
        }
    }

    /// The info sidebar: property table, channel toggles, then the mipmap picker. Returns the new
    /// (level, channels) if either changed.
    /// Whether this preview has anything for the Details panel.
    pub fn has_details(&self) -> bool {
        match self {
            Self::Image { .. } => true,
            Self::Material(material) => material.has_params(),
            Self::Model(_) => true,
            Self::Uld(layout) => layout.has_details(),
            Self::Font(_)
            | Self::Icons(_)
            | Self::Luab(_)
            | Self::Shpk(_)
            | Self::Shcd(_)
            | Self::Imc(_)
            | Self::Stm(_)
            | Self::Atch(_)
            | Self::Avfx(_)
            | Self::Eid(_)
            | Self::Est(_)
            | Self::Exl(_)
            | Self::Skp(_)
            | Self::Tera(_)
            | Self::Tmb(_)
            | Self::Pap(_)
            | Self::Phyb(_)
            | Self::Sklb(_)
            | Self::Eqdp(_)
            | Self::Eqp(_)
            | Self::Gmp(_)
            | Self::Evp(_)
            | Self::Layers(_)
            | Self::Zone(_)
            | Self::Environments(_)
            | Self::Ambient(_)
            | Self::Spm(_)
            | Self::Pbd(_)
            | Self::Pcb(_)
            | Self::Cmp(_)
            | Self::GrassZone(_)
            | Self::GrassGrid(_)
            | Self::Dic(_)
            | Self::Fpeb(_)
            | Self::Waoe(_)
            | Self::Wtd(_)
            | Self::Cutb(_)
            | Self::Scd(_) => true,
            _ => false,
        }
    }

    /// `view` is what the picker below the table is currently set to, and what it returns is the
    /// same triple once the user has moved it.
    pub fn info_ui(
        &self,
        ui: &mut egui::Ui,
        view: (u8, u16, Channels),
        follow: &mut Option<String>,
        deps: &mut crate::assets::deps::Deps,
        backend: &crate::backend::Backend,
    ) -> Option<(u8, u16, Channels)> {
        if let Self::Material(material) = self {
            material::details_ui(ui, material, follow);
            return None;
        }
        if let Self::Model(model) = self {
            model.details_ui(ui, follow);
            return None;
        }
        if let Self::Uld(layout) = self {
            uld::details_ui(ui, layout, deps, backend);
            return None;
        }
        if let Self::Font(font) = self {
            font.details_ui(ui, follow);
            return None;
        }
        if let Self::Icons(icons) = self {
            icons.details_ui(ui, follow);
            return None;
        }
        if let Self::Luab(chunk) = self {
            chunk.details_ui(ui);
            return None;
        }
        if let Self::Shpk(package) = self {
            package.details_ui(ui);
            return None;
        }
        if let Self::Shcd(code) = self {
            code.details_ui(ui);
            return None;
        }
        if let Self::Imc(change) = self {
            change.details_ui(ui);
            return None;
        }
        if let Self::Stm(templates) = self {
            templates.details_ui(ui);
            return None;
        }
        if let Self::Atch(points) = self {
            points.details_ui(ui);
            return None;
        }
        if let Self::Avfx(effect) = self {
            effect.details_ui(ui, follow);
            return None;
        }
        if let Self::Eid(points) = self {
            points.details_ui(ui);
            return None;
        }
        if let Self::Est(templates) = self {
            templates.details_ui(ui);
            return None;
        }
        if let Self::Exl(list) = self {
            list.details_ui(ui);
            return None;
        }
        if let Self::Skp(parameters) = self {
            parameters.details_ui(ui);
            return None;
        }
        if let Self::Tera(terrain) = self {
            terrain.details_ui(ui);
            return None;
        }
        if let Self::Tmb(timeline) = self {
            timeline.details_ui(ui);
            return None;
        }
        if let Self::Pap(pack) = self {
            pack.details_ui(ui);
            return None;
        }
        if let Self::Phyb(physics) = self {
            physics.details_ui(ui);
            return None;
        }
        if let Self::Sklb(skeleton) = self {
            skeleton.details_ui(ui);
            return None;
        }
        if let Self::Eqdp(parameters) = self {
            parameters.details_ui(ui);
            return None;
        }
        if let Self::Eqp(parameters) = self {
            parameters.details_ui(ui);
            return None;
        }
        if let Self::Gmp(parameters) = self {
            parameters.details_ui(ui);
            return None;
        }
        if let Self::Evp(parameters) = self {
            parameters.details_ui(ui);
            return None;
        }
        if let Self::Layers(layers) = self {
            layers.details_ui(ui, follow, deps, backend);
            return None;
        }
        if let Self::Zone(annotations) = self {
            annotations.details_ui(ui);
            return None;
        }
        if let Self::Environments(set) = self {
            set.details_ui(ui, follow);
            return None;
        }
        if let Self::Ambient(light) = self {
            light.details_ui(ui);
            return None;
        }
        if let Self::Spm(parameters) = self {
            parameters.details_ui(ui);
            return None;
        }
        if let Self::Pbd(deformers) = self {
            deformers.details_ui(ui);
            return None;
        }
        if let Self::Pcb(collision) = self {
            collision.details_ui(ui);
            return None;
        }
        if let Self::Cmp(parameters) = self {
            parameters.details_ui(ui, deps, backend);
            return None;
        }
        if let Self::Fpeb(held) = self {
            held.details_ui(ui);
            return None;
        }
        if let Self::Waoe(held) = self {
            held.details_ui(ui);
            return None;
        }
        if let Self::Wtd(held) = self {
            held.details_ui(ui);
            return None;
        }
        if let Self::Dic(dictionary) = self {
            dictionary.details_ui(ui);
            return None;
        }
        if let Self::Cutb(cutscene) = self {
            cutscene.details_ui(ui);
            return None;
        }
        if let Self::Scd(container) = self {
            container.details_ui(ui);
            return None;
        }
        if let Self::GrassZone(zone) = self {
            zone.details_ui(ui);
            return None;
        }
        if let Self::GrassGrid(grid) = self {
            grid.details_ui(ui);
            return None;
        }
        let Self::Image {
            facts,
            mips,
            depth,
            components,
            ..
        } = self
        else {
            return None;
        };
        let (mip, mut slice, mut channels) = view;
        let mut level = mip;
        ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
            egui::Grid::new("asset_facts")
                .num_columns(2)
                .striped(true)
                .show(ui, |ui| {
                    for (label, value) in facts {
                        ui.label(RichText::new(*label).weak());
                        ui.label(value);
                        // Stripes are drawn across the summed column widths, so the last column has
                        // to take the slack or they stop short of the panel edge.
                        ui.allocate_space(Vec2::new(ui.available_width(), 0.0));
                        ui.end_row();
                    }
                });

            if *components > 1 {
                ui.add_space(8.0);
                ui.separator();
                ui.label(RichText::new("Channels").weak());
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    // A single-component format is drawn as gray, so it has no toggles at all.
                    let offered: &mut [(&str, &mut bool)] = match components {
                        2 => &mut [("R", &mut channels.r), ("G", &mut channels.g)],
                        3 => &mut [
                            ("R", &mut channels.r),
                            ("G", &mut channels.g),
                            ("B", &mut channels.b),
                        ],
                        _ => &mut [
                            ("R", &mut channels.r),
                            ("G", &mut channels.g),
                            ("B", &mut channels.b),
                            ("A", &mut channels.a),
                        ],
                    };
                    for (name, flag) in offered {
                        ui.toggle_value(flag, *name);
                    }
                });
            }

            if *depth > 1 {
                ui.add_space(8.0);
                ui.separator();
                ui.label(RichText::new(if *depth == 6 { "Face" } else { "Slice" }).weak());
                ui.add_space(4.0);
                ui.add(egui::Slider::new(&mut slice, 0..=depth.saturating_sub(1)));
            }

            if mips.len() > 1 {
                ui.add_space(8.0);
                ui.separator();
                ui.label(RichText::new("Mipmaps").weak());
                ui.add_space(4.0);
                for entry in mips {
                    let label = format!(
                        "{}: {} x {}  ({})",
                        entry.level,
                        entry.width,
                        entry.height,
                        Bytes(entry.bytes)
                    );
                    if ui.selectable_label(entry.level == mip, label).clicked() {
                        level = entry.level;
                    }
                }
            }
        });
        ((level, slice, channels) != view).then_some((level, slice, channels))
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Viewer {
    Texture,
    Image,
    Material,
    Model,
    Uld,
    Font,
    Icons,
    Luab,
    Shpk,
    Shcd,
    Imc,
    Stm,
    Atch,
    Avfx,
    Eid,
    Est,
    Exl,
    Skp,
    Tera,
    Tmb,
    Pap,
    Phyb,
    Sklb,
    Eqdp,
    Eqp,
    Gmp,
    Evp,
    Hwc,
    Lgb,
    Sgb,
    Lvb,
    Svb,
    Lcb,
    Uwb,
    Envb,
    Obsb,
    Essb,
    Amb,
    Spm,
    Pbd,
    Pcb,
    Cmp,
    Gzd,
    Ggd,
    Dic,
    Fpeb,
    Waoe,
    Wtd,
    Cutb,
    Scd,
    Text,
    Raw,
}

impl Viewer {
    /// Everything except `Raw`, which the dropdown offers separately. Fixed order, so a given
    /// viewer sits in the same place whatever file is selected.
    pub const RENDERED: [Self; 51] = [
        Self::Texture,
        Self::Image,
        Self::Material,
        Self::Model,
        Self::Uld,
        Self::Font,
        Self::Icons,
        Self::Luab,
        Self::Shpk,
        Self::Shcd,
        Self::Imc,
        Self::Stm,
        Self::Atch,
        Self::Avfx,
        Self::Eid,
        Self::Est,
        Self::Exl,
        Self::Skp,
        Self::Tera,
        Self::Tmb,
        Self::Pap,
        Self::Phyb,
        Self::Sklb,
        Self::Eqdp,
        Self::Eqp,
        Self::Gmp,
        Self::Evp,
        Self::Hwc,
        Self::Lgb,
        Self::Sgb,
        Self::Lvb,
        Self::Svb,
        Self::Lcb,
        Self::Uwb,
        Self::Envb,
        Self::Obsb,
        Self::Essb,
        Self::Amb,
        Self::Spm,
        Self::Pbd,
        Self::Pcb,
        Self::Cmp,
        Self::Gzd,
        Self::Ggd,
        Self::Dic,
        Self::Fpeb,
        Self::Waoe,
        Self::Wtd,
        Self::Cutb,
        Self::Scd,
        Self::Text,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Texture => "Texture",
            Self::Image => "Image",
            Self::Material => "Material",
            Self::Model => "Model",
            Self::Uld => "Layout",
            Self::Font => "Font",
            Self::Icons => "Icons",
            Self::Luab => "Lua",
            Self::Shpk => "Shader package",
            Self::Shcd => "Shader code",
            Self::Imc => "Image change",
            Self::Stm => "Staining template",
            Self::Atch => "Attach points",
            Self::Avfx => "Visual effect",
            Self::Eid => "Bind points",
            Self::Est => "Skeleton template",
            Self::Exl => "Sheet list",
            Self::Skp => "Skeleton parameters",
            Self::Tera => "Terrain",
            Self::Tmb => "Timeline",
            Self::Pap => "Animation",
            Self::Phyb => "Physics",
            Self::Sklb => "Skeleton",
            Self::Eqdp => "Deformer parameters",
            Self::Eqp => "Equipment parameters",
            Self::Gmp => "Gimmick parameters",
            Self::Evp => "Equipment VFX parameters",
            Self::Hwc => "Cursor",
            Self::Lgb => "Layer group",
            Self::Sgb => "Shared group",
            Self::Lvb => "Level",
            Self::Svb => "Sky visibility",
            Self::Lcb => "Light culling",
            Self::Uwb => "Underwater",
            Self::Envb => "Environment",
            Self::Obsb => "Object behavior",
            Self::Essb => "Environment sound",
            Self::Amb => "Ambient light",
            Self::Spm => "Shader parameters",
            Self::Pbd => "Bone deformers",
            Self::Pcb => "Collision",
            Self::Cmp => "Character make",
            Self::Gzd => "Grass zone",
            Self::Ggd => "Grass grid",
            Self::Dic => "Word dictionary",
            Self::Fpeb => "Facial parameter edits",
            Self::Waoe => "Attach offset list",
            Self::Wtd => "Weapon type table",
            Self::Cutb => "Cutscene",
            Self::Scd => "Sound",
            Self::Text => "Text",
            Self::Raw => "Bytes",
        }
    }

    /// The extensions this viewer reads.
    pub fn extensions(self) -> impl Iterator<Item = &'static str> {
        super::EXTENSIONS
            .iter()
            .filter(move |(_, _, viewer)| *viewer == self)
            .map(|(extension, ..)| *extension)
    }

    /// The label with the extensions it reads, for the dropdown.
    pub fn described(self) -> String {
        let extensions = self.extensions().collect::<Vec<_>>();
        match extensions.is_empty() {
            true => self.label().to_owned(),
            false => format!("{} ({})", self.label(), extensions.join(", ")),
        }
    }

    /// What a path's name says it holds. An unnamed file has nothing here to go on.
    pub fn from_extension(path: &str) -> Self {
        let extension = path.rsplit('.').next().unwrap_or_default();
        super::EXTENSIONS
            .iter()
            .find(|(name, ..)| *name == extension)
            .map_or(Self::Raw, |(.., viewer)| *viewer)
    }
}

/// One mipmap level, for the picker under the info table.
pub struct Mip {
    pub level: u8,
    pub width: u16,
    pub height: u16,
    pub bytes: usize,
}

/// Beyond the raw file: an exact DDS container, and a lossless PNG (zipped, past one face, layer
/// or slice) where the format has one. Decoded fresh at click time rather than kept: `Preview`
/// only holds a `.tex`'s uploaded texture handle, not the container `dds`/`png` read from.
fn texture_export_choices<'a>(bytes: &'a [u8], path: &str, mip: u8) -> Vec<export::Choice<'a>> {
    use ironworks::file::{File as _, tex};

    let path = path.to_owned();
    vec![
        export::Choice::bytes("DDS", "texture.dds", move || {
            let texture = tex::Texture::read(std::io::Cursor::new(bytes.to_vec()))?;
            crate::utils::dds(&texture)
        })
        .hover("The file's own pixel blocks, exact, with every mip level")
        .filter("DDS image", &["dds"]),
        export::Choice::named_bytes("PNG", move || {
            let texture = tex::Texture::read(std::io::Cursor::new(bytes.to_vec()))?;
            let level = mip.min(texture.mip_levels().saturating_sub(1));
            let images = crate::utils::png(&texture, level, &path)?.ok_or_else(|| {
                anyhow::anyhow!(
                    "{:?} has no lossless PNG form; use DDS instead",
                    texture.format()
                )
            })?;
            Ok((images.file_name("texture"), images.bytes()))
        })
        .hover(
            "The mip level shown now; zipped when the texture has more than one face, layer or \
             slice",
        ),
    ]
}

/// Build the image preview both the image and texture viewers end at.
#[allow(clippy::too_many_arguments)]
pub(super) fn upload(
    ctx: &egui::Context,
    path: &str,
    image: image::DynamicImage,
    depth: u16,
    components: u8,
    facts: Vec<(&'static str, String)>,
    mips: Vec<Mip>,
    channels: Channels,
) -> Preview {
    let mut rgba = image.to_rgba8();
    channels.apply(&mut rgba);
    let size = [rgba.width() as usize, rgba.height() as usize];
    // Sized from the source rather than from what was uploaded, so a texture the renderer made us
    // scale down still draws at the dimensions the file states.
    let held = crate::utils::tex_loader::fit(ctx, &rgba);
    let texture = ctx.load_texture(
        format!("asset:{path}"),
        egui::ColorImage::from_rgba_unmultiplied(
            [held.width() as usize, held.height() as usize],
            held.as_flat_samples().as_slice(),
        ),
        // Nearest keeps a zoomed-in mipmap readable as actual texels rather than a blur.
        egui::TextureOptions::NEAREST,
    );
    Preview::Image {
        texture,
        size,
        depth,
        components,
        facts,
        mips,
    }
}

#[cfg(test)]
mod tests {
    use super::{TABLE, table, table_id};

    /// The header is painted at the offset the rows were left at, which only works while the id it
    /// reads is the one the scroll area wrote.
    #[test]
    fn the_header_reads_the_offset_the_rows_kept() {
        let mut id = None;
        egui::__run_test_ui(|ui| {
            id = Some(table_id(ui));
            table(ui, &[("Column", 8)], 4, |ui, index| {
                ui.label(index.to_string());
            });
            assert!(
                egui::scroll_area::State::load(ui.ctx(), id.unwrap()).is_some(),
                "the table's scroll area is somewhere else"
            );
        });
        // And the bare string is not it, which is what the salting is for.
        egui::__run_test_ui(|ui| {
            assert_ne!(ui.make_persistent_id(TABLE), id.unwrap());
        });
    }
}
