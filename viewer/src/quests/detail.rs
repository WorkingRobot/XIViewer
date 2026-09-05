use std::collections::HashMap;

use anyhow::{Result, anyhow};
use compact_str::ToCompactString;
use egui::{CollapsingHeader, Color32, Label, RichText, ScrollArea, Sense};
use ironworks::{excel::Language, sestring::SeStr};

use crate::{
    backend::Backend,
    excel::{
        base::CachedProvider,
        provider::{ExcelProvider, ExcelRow, ExcelSheet},
    },
    quests::{
        Load,
        derive::{self, Param},
        index::Index,
        requirements,
        rewards::{self, Catalog},
        script::{self, Script},
    },
    settings::EVALUATE_STRINGS,
    sheet::{CellResponse, SheetColumnDefinition},
};

const IDENTITY: &[&str] = &[
    "Id",
    "Expansion",
    "PlaceName",
    "JournalGenre",
    "SortKey",
    "Icon",
    "IconSpecial",
    "EventIconType",
];

/// Requirement fields `requirements::ui` does not resolve to something more specific, shown plainly
/// by the generic `fields()` renderer.
const REQUIREMENTS: &[&str] = &[
    "LevelMax",
    "QuestLevelOffset",
    "ClassJobRequired",
    "ClassJobUnlock",
    "Festival",
];

const FLOW: &[&str] = &[
    "IssuerStart",
    "IssuerLocation",
    "TargetEnd",
    "InstanceContent[0]",
    "InstanceContent[1]",
    "InstanceContent[2]",
    "InstanceContentUnlock",
    "DeliveryQuest",
    "SatisfactionNpc",
    "SatisfactionLevel",
    "IsRepeatable",
    "RepeatIntervalType",
    "QuestRepeatFlag",
    "DailyQuestPool",
    "CanCancel",
    "Introduction",
    "HideOfferIcon",
    "HideInScenarioGuide",
];

/// `SystemReward` and `GCTypeReward` name no sheet EXDSchema or the corpus can confirm, so they fall
/// back to the raw-field renderer rather than a resolved one.
const UNRESOLVED_REWARDS: &[&str] = &["GCTypeReward", "SystemReward[0]", "SystemReward[1]"];

/// One line of a quest's dialogue, by the row key the script names it with.
pub struct Line {
    pub key: String,
    pub speaker: String,
    pub text: Vec<u8>,
}

pub struct Dialogue {
    lines: Vec<Line>,
    /// The three fixed buckets first, then one group per speaker in the order they first talk, as
    /// indices into [`Self::lines`].
    groups: Vec<(String, Vec<usize>)>,
    by_key: HashMap<String, usize>,
}

impl Dialogue {
    pub fn line(&self, key: &str) -> Option<&Line> {
        self.by_key.get(key).map(|at| &self.lines[*at])
    }
}

pub struct Links {
    script: String,
    /// The instruction a script reads the asset by, and the file it names.
    music: Vec<(String, String)>,
    cutscenes: Vec<(String, String)>,
}

impl Links {
    /// The file a `QuestParams` instruction names, for a script reading `self.<param>`.
    pub fn asset(&self, param: &str) -> Option<&str> {
        self.music
            .iter()
            .chain(&self.cutscenes)
            .find(|(name, _)| name == param)
            .map(|(_, path)| path.as_str())
    }
}

#[derive(Default)]
pub struct Detail {
    node: Option<u32>,
    dialogue: Load<Dialogue>,
    links: Load<Links>,
    script: Load<Script>,
    /// The reward catalog is keyed by language, not by quest, so it survives a selection change.
    catalog: Load<Catalog>,
    catalog_for: Option<Language>,
}

pub enum Action {
    Select(u32),
    Navigate(String),
}

impl Detail {
    /// Dialogue and the asset links are per quest and neither is cheap, so both wait for a
    /// selection rather than being built with the index.
    pub fn poll(&mut self, backend: &Backend, index: &Index, node: u32, language: Language) {
        if self.node != Some(node) {
            self.node = Some(node);
            self.dialogue = Load::Idle;
            self.links = Load::Idle;
            self.script = Load::Idle;
        }
        let quest = index.quest(node);
        if matches!(self.dialogue, Load::Idle) {
            let excel = backend.excel().clone();
            let name = derive::text_sheet(quest.row_id, &quest.id);
            let id = quest.id.to_uppercase();
            self.dialogue = Load::spawn(async move { dialogue(excel, name, id, language).await });
        }
        if matches!(self.links, Load::Idle) {
            let backend = backend.clone();
            let script = derive::script_path(quest.row_id, &quest.id);
            let params = index.assets(node);
            self.links = Load::spawn(async move { links(backend, language, script, params).await });
        }
        if matches!(self.script, Load::Idle) {
            let files = backend.files().clone();
            let path = derive::script_path(quest.row_id, &quest.id);
            self.script = Load::spawn(async move { script::read(&files.read(&path).await?) });
        }
        if self.catalog_for != Some(language) {
            self.catalog_for = Some(language);
            self.catalog = Load::Idle;
        }
        if matches!(self.catalog, Load::Idle) {
            let backend = backend.clone();
            self.catalog = Load::spawn(async move { Catalog::load(backend, language).await });
        }
        self.dialogue.poll();
        self.links.poll();
        self.script.poll();
        self.catalog.poll();
    }

    /// The quest's dialogue, its assets and its scenes, each once the read finishes.
    pub fn dialogue_lines(&self) -> Option<&Dialogue> {
        match &self.dialogue {
            Load::Ready(held) => Some(held),
            _ => None,
        }
    }

    pub fn links(&self) -> Option<&Links> {
        match &self.links {
            Load::Ready(held) => Some(held),
            _ => None,
        }
    }

    pub fn script(&self) -> &Load<Script> {
        &self.script
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, index: &Index, node: u32) -> Option<Action> {
        let mut action = None;
        ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
            let quest = index.quest(node);
            ui.label(
                RichText::new(format!("{} · row {}", quest.id, quest.row_id))
                    .weak()
                    .small(),
            );
            ui.add_space(6.0);
            action = self.body(ui, index, node);
        });
        action
    }

    fn body(&mut self, ui: &mut egui::Ui, index: &Index, node: u32) -> Option<Action> {
        let Some(row) = index.row(node) else {
            ui.colored_label(Color32::RED, "The quest's row went away");
            return None;
        };
        let mut action = fields(
            ui,
            index,
            row,
            IDENTITY.iter().map(|name| (*name).to_string()),
        );

        action = action.or(section(ui, "Requirements", false, |ui| {
            let mut action = requirements::ui(ui, index, row);
            action = action.or(fields(
                ui,
                index,
                row,
                REQUIREMENTS.iter().map(|n| (*n).to_string()),
            ));
            action
        }));
        action = action.or(section(ui, "Progression", false, |ui| {
            fields(ui, index, row, FLOW.iter().map(|n| (*n).to_string()))
        }));
        action = action.or(section(ui, "Rewards", true, |ui| {
            let mut action = match &self.catalog {
                Load::Ready(catalog) => rewards::ui(ui, index, row, catalog),
                Load::Failed(error) => {
                    ui.colored_label(Color32::RED, error.clone());
                    None
                }
                Load::Idle | Load::Loading(_) => {
                    ui.spinner();
                    None
                }
            };
            action = action.or(fields(
                ui,
                index,
                row,
                UNRESOLVED_REWARDS.iter().map(|n| (*n).to_string()),
            ));
            action
        }));

        action = action.or(self.relations(ui, index, node));
        action = action.or(self.files(ui));
        action = action.or(self.dialogue(ui));
        action
    }

    fn relations(&self, ui: &mut egui::Ui, index: &Index, node: u32) -> Option<Action> {
        let mut action = None;
        let prereqs = index.graph.prereqs(node);
        if !prereqs.is_empty() {
            let any = index.quest(node).join == 2 && prereqs.len() > 1;
            action = action.or(section(ui, "Requires", true, |ui| {
                if any {
                    ui.label(RichText::new("Any one of these will do.").weak().small());
                }
                quest_list(ui, index, prereqs)
            }));
        }
        let dependents = index.graph.dependents(node);
        if !dependents.is_empty() {
            action = action.or(section(ui, "Unlocks", true, |ui| {
                quest_list(ui, index, dependents)
            }));
        }
        let locks: Vec<u32> = index
            .quest(node)
            .lock
            .iter()
            .filter_map(|row_id| index.node_of(*row_id))
            .collect();
        if !locks.is_empty() {
            action = action.or(section(ui, "Alternatives", true, |ui| {
                ui.label(
                    RichText::new("Taking one of these puts the others out of reach.")
                        .weak()
                        .small(),
                );
                quest_list(ui, index, &locks)
            }));
        }
        action
    }

    fn files(&self, ui: &mut egui::Ui) -> Option<Action> {
        let title = match &self.links {
            Load::Ready(links) => {
                format!("Files ({})", 1 + links.music.len() + links.cutscenes.len())
            }
            _ => "Files".to_string(),
        };
        section(ui, &title, true, |ui| match &self.links {
            Load::Idle | Load::Loading(_) => {
                ui.spinner();
                None
            }
            Load::Failed(error) => {
                ui.colored_label(Color32::RED, error.clone());
                None
            }
            Load::Ready(links) => {
                let mut action = asset_link(ui, &links.script);
                for (_, path) in links.music.iter().chain(&links.cutscenes) {
                    action = action.or(asset_link(ui, path));
                }
                action
            }
        })
    }

    fn dialogue(&self, ui: &mut egui::Ui) -> Option<Action> {
        let title = match &self.dialogue {
            Load::Ready(dialogue) => format!("Dialogue ({})", dialogue.lines.len()),
            _ => "Dialogue".to_string(),
        };
        section(ui, &title, false, |ui| {
            match &self.dialogue {
                Load::Idle | Load::Loading(_) => {
                    ui.spinner();
                }
                Load::Failed(error) => {
                    ui.colored_label(Color32::RED, error.clone());
                }
                Load::Ready(dialogue) => {
                    for (speaker, held) in &dialogue.groups {
                        ui.add_space(4.0);
                        ui.label(RichText::new(speaker).strong());
                        for at in held {
                            ui.label(sestring(ui, &dialogue.lines[*at].text));
                        }
                    }
                }
            }
            None
        })
    }
}

/// A collapsing section that only exists when its body drew something.
fn section(
    ui: &mut egui::Ui,
    title: &str,
    open: bool,
    body: impl FnOnce(&mut egui::Ui) -> Option<Action>,
) -> Option<Action> {
    CollapsingHeader::new(title)
        .id_salt(title)
        .default_open(open)
        .show(ui, body)
        .body_returned
        .flatten()
}

/// A `CellResponse` as the `Action` a click on it should take, the way every generic field row
/// already resolves one.
pub(crate) fn link_action(response: CellResponse) -> Option<Action> {
    match response {
        CellResponse::Link((sheet, (row_id, subrow))) => Some(Action::Navigate(match subrow {
            Some(subrow) => format!("/sheet/{sheet}#R{row_id}.{subrow}"),
            None => format!("/sheet/{sheet}#R{row_id}"),
        })),
        _ => None,
    }
}

fn fields(
    ui: &mut egui::Ui,
    index: &Index,
    row: ExcelRow<'_>,
    names: impl Iterator<Item = String>,
) -> Option<Action> {
    let mut action = None;
    for name in names {
        let Some(at) = index.column(&name) else {
            continue;
        };
        let Ok((_, column)) = index.table.get_column_by_offset(at) else {
            continue;
        };
        if blank(&name, row, column) {
            continue;
        }
        let Ok(cell) = index.table.cell_by_offset(row, at) else {
            continue;
        };
        ui.horizontal(|ui| {
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
            ui.add_sized(
                [ui.available_width() * 0.45, ui.spacing().interact_size.y],
                Label::new(RichText::new(&name).weak()),
            )
            .on_hover_text(&name);
            if let Some(new_action) = link_action(cell.show(ui).inner) {
                action = Some(new_action);
            }
        });
    }
    action
}

/// Rows the game leaves at zero or empty are slots the quest does not use.
///
/// `BeastReputationValue` also uses `0xFFFF`: the game reads it only when `BeastTribe` is set, and
/// only to cap the rank's own required reputation downward, so `0xFFFF` means "no cap" rather than
/// a required amount.
fn blank(name: &str, row: ExcelRow<'_>, column: &SheetColumnDefinition) -> bool {
    if column.kind() == ironworks::file::exh::ColumnKind::String {
        return row
            .read_string(u32::from(column.offset()))
            .is_ok_and(|value| value.as_bytes().is_empty());
    }
    let Ok(value) =
        crate::sheet::read_integer::<i64>(row, u32::from(column.offset()), column.kind())
    else {
        return false;
    };
    value == 0 || (name == "BeastReputationValue" && value == 0xFFFF)
}

fn quest_list(ui: &mut egui::Ui, index: &Index, nodes: &[u32]) -> Option<Action> {
    let mut action = None;
    for node in nodes {
        let quest = index.quest(*node);
        let response = ui
            .add(
                Label::new(RichText::new(&quest.name).color(ui.visuals().hyperlink_color))
                    .truncate()
                    .sense(Sense::click()),
            )
            .on_hover_text(&quest.id)
            .on_hover_cursor(egui::CursorIcon::PointingHand);
        if response.clicked() {
            action = Some(Action::Select(quest.row_id));
        }
    }
    action
}

/// A path link showing the file's own name, with the full path on hover, matching the convention
/// for reference-path links elsewhere in the app.
pub fn path_link(ui: &mut egui::Ui, path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    let response = ui
        .add(
            Label::new(
                RichText::new(name)
                    .color(ui.visuals().hyperlink_color)
                    .small(),
            )
            .truncate()
            .show_tooltip_when_elided(false)
            .sense(Sense::click()),
        )
        .on_hover_text(path)
        .on_hover_cursor(egui::CursorIcon::PointingHand);
    response.clicked()
}

fn asset_link(ui: &mut egui::Ui, path: &str) -> Option<Action> {
    path_link(ui, path).then(|| Action::Navigate(format!("/assets/{path}")))
}

/// Dialogue leans on more payload kinds than most sheets, so a player-name macro reads as a gap in
/// the sentence. That is the formatter having nothing to put there, not a decode failure.
pub fn sestring(ui: &egui::Ui, bytes: &[u8]) -> String {
    let text: &SeStr = bytes.into();
    if EVALUATE_STRINGS.get(ui.ctx()) {
        text.format()
            .try_to_compact_string()
            .map_or_else(|_| String::new(), Into::into)
    } else {
        text.macro_string()
            .try_to_compact_string()
            .map_or_else(|_| String::new(), Into::into)
    }
}

async fn dialogue(
    excel: CachedProvider,
    name: String,
    id_upper: String,
    language: Language,
) -> Result<Dialogue> {
    let sheet = excel.get_sheet(&name, language).await?;
    let columns = SheetColumnDefinition::from_sheet(&sheet);
    let (key, body) = match columns.as_slice() {
        [key, body, ..] => (key, body),
        _ => return Err(anyhow!("{name} is not a two column text sheet")),
    };

    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    let mut lines: Vec<Line> = Vec::new();
    let mut by_key = HashMap::new();
    for row_id in sheet.get_row_ids() {
        let Ok(row) = sheet.get_row(row_id) else {
            continue;
        };
        let (Ok(key), Ok(text)) = (
            row.read_string(u32::from(key.offset())),
            row.read_string(u32::from(body.offset())),
        ) else {
            continue;
        };
        if text.as_bytes().is_empty() {
            continue;
        }
        let key = key.format().to_string();
        let speaker = match derive::line_of(&key, &id_upper) {
            derive::Line::Journal => "Journal".to_string(),
            derive::Line::Objective => "Objectives".to_string(),
            derive::Line::System => "System".to_string(),
            derive::Line::Speaker(speaker) => speaker.to_string(),
        };
        let at = lines.len();
        by_key.insert(key.clone(), at);
        match groups.iter_mut().find(|(name, _)| *name == speaker) {
            Some((_, held)) => held.push(at),
            None => groups.push((speaker.clone(), vec![at])),
        }
        lines.push(Line {
            key,
            speaker,
            text: text.as_bytes().to_vec(),
        });
    }
    Ok(Dialogue {
        lines,
        groups,
        by_key,
    })
}

async fn links(
    backend: Backend,
    language: Language,
    script: String,
    params: Vec<(String, Param, u32)>,
) -> Result<Links> {
    let mut music = Vec::new();
    let mut cutscenes = Vec::new();
    for (instruction, param, arg) in params {
        let name = match param {
            Param::Bgm => "BGM",
            Param::Cutscene => "Cutscene",
        };
        let sheet = backend.excel().get_sheet(name, language).await?;
        // Both sheets name their file in their first schema field, which is the first column in
        // offset order.
        let Some(column) = SheetColumnDefinition::from_sheet(&sheet).into_iter().next() else {
            continue;
        };
        let Ok(row) = sheet.get_row(arg) else {
            continue;
        };
        let Ok(value) = row.read_string(u32::from(column.offset())) else {
            continue;
        };
        let value = value.format().to_string();
        if value.is_empty() {
            continue;
        }
        match param {
            Param::Bgm => music.push((instruction, value)),
            Param::Cutscene => cutscenes.push((instruction, derive::cutscene_path(&value))),
        }
    }
    music.sort_unstable();
    music.dedup();
    cutscenes.sort_unstable();
    cutscenes.dedup();

    // An instruction name can carry a cutscene id in a namespace of its own, so a link is only
    // offered for a file that is really there. A missing one means "not shown", not "absent".
    let paths: Vec<String> = cutscenes.iter().map(|(_, path)| path.clone()).collect();
    let present = backend.files().exists_many(&paths).await?;
    let cutscenes = cutscenes
        .into_iter()
        .zip(present)
        .filter_map(|(held, exists)| exists.then_some(held))
        .collect();

    Ok(Links {
        script,
        music,
        cutscenes,
    })
}
