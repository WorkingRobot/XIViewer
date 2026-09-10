use egui::{Frame, Layout, Modal, Sense, TextEdit, UiBuilder, Vec2, WidgetText};

use crate::{
    DEFAULT_API_URL,
    backend::Backend,
    data::web::{VersionInfo, WebFileProvider},
    github::GithubApi,
    schema::web::WebProvider,
    settings::{
        BACKEND_CONFIG, BackendConfig, GithubSchemaBranch, GithubSchemaLocation, InstallLocation,
        Region, SchemaLocation,
    },
    utils::{ConvertiblePromise, PromiseKind, TrackedPromise, UnsendPromise},
};

#[cfg(target_arch = "wasm32")]
use crate::worker::WorkerDirectory;

type VersionPromise<T> = ConvertiblePromise<TrackedPromise<anyhow::Result<T>>, Option<T>>;
type VersionPromiseHolder<K, T> = Option<(K, VersionPromise<T>)>;

pub struct SetupWindow {
    api_url: String,
    location: InstallLocation,
    schema: SchemaLocation,
    is_startup: bool,
    #[cfg(target_arch = "wasm32")]
    location_promises: SetupPromises,
    #[cfg(target_arch = "wasm32")]
    schema_promises: SetupPromises,
    setup_promise: Option<UnsendPromise<anyhow::Result<(Backend, BackendConfig)>>>,
    display_error: Option<anyhow::Error>,

    web_version_promise: VersionPromiseHolder<(String, Region), VersionInfo>,
    web_regions_promise: VersionPromiseHolder<String, Vec<String>>,
    /// Keyed on (owner, repo, signed in)
    github_branch_promise: VersionPromiseHolder<(String, String, bool), Vec<GithubSchemaBranch>>,
}

impl SetupWindow {
    pub fn from_blank(is_startup: bool) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let location = ironworks::sqpack::Install::search()
            .and_then(|p| Some(InstallLocation::Sqpack(p.path().to_str()?.to_owned())))
            .unwrap_or(InstallLocation::Web(Region::Global, None));

        #[cfg(target_arch = "wasm32")]
        let location = InstallLocation::Web(Region::Global, None);

        Self {
            api_url: DEFAULT_API_URL.to_string(),
            location,
            schema: SchemaLocation::Github(GithubSchemaLocation {
                owner: super::DEFAULT_GITHUB_REPO.0.to_string(),
                repo: super::DEFAULT_GITHUB_REPO.1.to_string(),
                branch: GithubSchemaBranch::Latest,
            }),
            is_startup,
            #[cfg(target_arch = "wasm32")]
            location_promises: Default::default(),
            #[cfg(target_arch = "wasm32")]
            schema_promises: Default::default(),
            setup_promise: None,
            display_error: None,
            web_version_promise: None,
            web_regions_promise: None,
            github_branch_promise: None,
        }
    }

    pub fn from_config(ctx: &egui::Context, is_startup: bool) -> Self {
        if let Some(Some(config)) = BACKEND_CONFIG.try_get(ctx) {
            Self {
                api_url: config.api_url,
                location: config.location,
                schema: config.schema,
                is_startup,
                #[cfg(target_arch = "wasm32")]
                location_promises: Default::default(),
                #[cfg(target_arch = "wasm32")]
                schema_promises: Default::default(),
                setup_promise: None,
                display_error: None,
                web_version_promise: None,
                web_regions_promise: None,
                github_branch_promise: None,
            }
        } else {
            Self::from_blank(is_startup)
        }
    }

    pub fn draw(&mut self, ctx: &egui::Context) -> Option<(Backend, BackendConfig)> {
        let github_token = crate::github::token(ctx);

        #[cfg(target_arch = "wasm32")]
        {
            if let Some(handle) = self.location_promises.take_folder() {
                self.location = InstallLocation::Worker(handle.0.name());
            }

            if let Some(handle) = self.schema_promises.take_folder() {
                self.schema = SchemaLocation::Worker(handle.0.name());
            }
        }

        let show_inner = |ui: &mut egui::Ui| {
            ui.vertical_centered(|ui| {
                ui.heading("Setup");
            });
            ui.separator();

            let enabled: bool;
            match self.setup_promise.take().map(PromiseKind::try_take) {
                None => {
                    enabled = true;
                }
                Some(Err(promise)) => {
                    self.setup_promise = Some(promise);
                    enabled = false;
                    ui.label("Loading...");
                }
                Some(Ok(Ok(backend))) => {
                    return Some(backend);
                }
                Some(Ok(Err(err))) => {
                    log::error!("Setup Error: {err}");
                    self.display_error = Some(err);
                    enabled = true;
                }
            }

            if let Some(err) = &self.display_error {
                ui.label(err.to_string());
            } else {
                ui.label("Please select the location of the game files and schema.");
            }

            let is_go_clicked = ui
                .add_enabled_ui(enabled, |ui| {
                    // Outside the Location group: it feeds the path list and song metadata whatever
                    // the file source is, so it is not a property of that choice.
                    ui.horizontal(|ui| {
                        ui.label("API:");
                        ui.add(
                            TextEdit::singleline(&mut self.api_url)
                                .desired_width(ui.available_width()),
                        );
                    });

                    Frame::group(ui.style()).show(ui, |ui| {
                        ui.vertical_centered(|ui| {
                            ui.heading("Location");
                        });

                        ui.horizontal(|ui| {
                            ui.columns_const(|[col_0, col_1]| {
                                #[cfg(not(target_arch = "wasm32"))]
                                if radio(
                                    col_0,
                                    matches!(self.location, InstallLocation::Sqpack(_)),
                                    "Local",
                                ) {
                                    self.location = InstallLocation::Sqpack(
                                        std::env::current_dir()
                                            .ok()
                                            .and_then(|p| Some(p.to_str()?.to_string()))
                                            .unwrap_or("/".to_owned()),
                                    );
                                }
                                #[cfg(target_arch = "wasm32")]
                                if radio(
                                    col_0,
                                    matches!(self.location, InstallLocation::Worker(_)),
                                    "Local",
                                ) {
                                    self.location =
                                        InstallLocation::Worker("Select folder".to_string());
                                }
                                if radio(
                                    col_1,
                                    matches!(self.location, InstallLocation::Web(_, _)),
                                    "Web",
                                ) {
                                    self.location = InstallLocation::Web(Region::Global, None);
                                }
                            });
                        });

                        let api_url = self.api_url.clone();
                        match &mut self.location {
                            #[cfg(not(target_arch = "wasm32"))]
                            InstallLocation::Sqpack(path) => {
                                ui.horizontal(|ui| {
                                    ui.label("Path:");
                                    ui.with_layout(Layout::right_to_left(egui::Align::Min), |ui| {
                                        if ui.button("Browse").clicked()
                                            && let Some(picked_path) = rfd::FileDialog::new()
                                                .pick_folder()
                                                .and_then(|d| d.to_str().map(|s| s.to_owned()))
                                        {
                                            *path = picked_path;
                                        }
                                        ui.add(
                                            egui::TextEdit::singleline(path)
                                                .desired_width(ui.available_width()),
                                        );
                                    });
                                });
                            }

                            #[cfg(target_arch = "wasm32")]
                            InstallLocation::Worker(name) => {
                                use crate::data::worker::WorkerFileProvider;
                                use web_sys::FileSystemPermissionMode;

                                if !*IS_DIRECTORY_PICKER_SUPPORTED {
                                    draw_unsupported_directory_picker(ui);
                                } else {
                                    ui.horizontal(|ui| {
                                        ui.label("Name:");
                                        ui.with_layout(
                                            Layout::right_to_left(egui::Align::Min),
                                            |ui| {
                                                if ui.button("Browse").clicked() {
                                                    self.location_promises.open_folder_picker(
                                                        FileSystemPermissionMode::Read,
                                                        WorkerFileProvider::add_folder,
                                                    );
                                                }
                                                egui::ComboBox::from_id_salt("install_folder")
                                                    .selected_text(name.as_str())
                                                    .width(ui.available_width())
                                                    .show_ui(ui, |ui| {
                                                        match self
                                                            .location_promises
                                                            .get_folder_list(
                                                                WorkerFileProvider::folders,
                                                            ) {
                                                            None => {
                                                                ui.label("Retrieving...");
                                                            }
                                                            Some(Err(e)) => {
                                                                ui.label(format!(
                                                                    "An error occured: {e}"
                                                                ));
                                                            }
                                                            Some(Ok(entries)) => {
                                                                if entries.is_empty() {
                                                                    ui.label("None");
                                                                } else {
                                                                    for entry in entries {
                                                                        ui.selectable_value(
                                                                            name,
                                                                            entry.0.name(),
                                                                            entry.0.name(),
                                                                        );
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    });
                                            },
                                        );
                                    });
                                }
                            }

                            InstallLocation::Web(region, version) => {
                                let url = &api_url;
                                // Fetch the list of available repositories (once per URL) to
                                // drive which regions can be selected.
                                if !url.is_empty()
                                    && self
                                        .web_regions_promise
                                        .as_ref()
                                        .is_none_or(|v| v.0 != *url)
                                {
                                    let repo_url = url.clone();
                                    self.web_regions_promise = Some((
                                        url.clone(),
                                        ConvertiblePromise::new_promise(
                                            TrackedPromise::spawn_local(async move {
                                                WebFileProvider::get_regions(&repo_url).await
                                            }),
                                        ),
                                    ));
                                }

                                // Which regions the backend serves. If the endpoint is missing,
                                // `None` leaves every region enabled rather than hiding them all.
                                let available_slugs: Option<Vec<String>> =
                                    if let Some((_, promise)) = &mut self.web_regions_promise {
                                        promise
                                            .get_mut(|r| match r {
                                                Ok(regions) => Some(regions),
                                                Err(e) => {
                                                    log::error!("Error fetching regions: {e}");
                                                    None
                                                }
                                            })
                                            .and_then(|repos| {
                                                repos.as_ref().map(|repos| repos.clone())
                                            })
                                    } else {
                                        None
                                    };

                                let is_region_available =
                                    |r: Region| r.is_available(available_slugs.as_deref());

                                ui.horizontal(|ui| {
                                    ui.label("Region:");
                                    egui::ComboBox::from_id_salt("setup_region")
                                        .selected_text(region.name())
                                        .width(ui.available_width())
                                        .show_ui(ui, |ui| {
                                            for r in [
                                                Region::Global,
                                                Region::Korea,
                                                Region::China,
                                                Region::Taiwan,
                                            ] {
                                                if is_region_available(r) {
                                                    ui.selectable_value(region, r, r.name());
                                                } else {
                                                    ui.add_enabled(
                                                        false,
                                                        egui::Button::selectable(
                                                            *region == r,
                                                            r.name(),
                                                        ),
                                                    );
                                                }
                                            }
                                        });
                                });

                                // (Re)fetch versions whenever the URL or region changes. On an
                                // actual change (not the initial load of a persisted config),
                                // reset the selected version so it can't dangle across regions.
                                let version_key = (url.clone(), *region);
                                let key_changed = self
                                    .web_version_promise
                                    .as_ref()
                                    .is_some_and(|v| v.0 != version_key);
                                if !url.is_empty()
                                    && region.is_available(available_slugs.as_deref())
                                    && self
                                        .web_version_promise
                                        .as_ref()
                                        .is_none_or(|v| v.0 != version_key)
                                {
                                    if key_changed {
                                        *version = None;
                                    }
                                    let ver_url = url.clone();
                                    let region_key = region.api_name().to_string();
                                    self.web_version_promise = Some((
                                        version_key,
                                        ConvertiblePromise::new_promise(
                                            TrackedPromise::spawn_local(async move {
                                                WebFileProvider::get_versions(&ver_url, &region_key)
                                                    .await
                                            }),
                                        ),
                                    ));
                                }

                                ui.horizontal(|ui| {
                                    ui.label("Version:");

                                    if let Some((_, promise)) = &mut self.web_version_promise {
                                        if let Some(versions) = promise.get_mut(|r| match r {
                                            Ok(vers) => {
                                                self.display_error = None;
                                                Some(vers)
                                            }
                                            Err(e) => {
                                                log::error!("Error fetching versions: {e}");
                                                self.display_error = Some(e);
                                                None
                                            }
                                        }) {
                                            if let Some(versions) = versions {
                                                egui::ComboBox::from_id_salt("setup_version")
                                                    .selected_text(version.as_ref().map_or_else(
                                                        || format!("Latest ({})", versions.latest),
                                                        |v| v.to_string(),
                                                    ))
                                                    .width(ui.available_width())
                                                    .show_ui(ui, |ui| {
                                                        version_row(
                                                            ui,
                                                            version,
                                                            None,
                                                            &format!(
                                                                "Latest ({})",
                                                                versions.latest
                                                            ),
                                                            versions.names.get(&versions.latest),
                                                        );
                                                        for entry in &versions.versions {
                                                            version_row(
                                                                ui,
                                                                version,
                                                                Some(entry.clone()),
                                                                &entry.to_string(),
                                                                versions.names.get(entry),
                                                            );
                                                        }
                                                    });
                                            } else {
                                                ui.label("Failed to load versions");
                                            }
                                        } else {
                                            ui.label("Loading versions...");
                                        }
                                    } else {
                                        ui.label("No versions available");
                                    }
                                });
                            }
                        }
                    });

                    let api_url = self.api_url.clone();
                    Frame::group(ui.style()).show(ui, |ui| {
                        ui.vertical_centered(|ui| {
                            ui.heading("Schema");
                        });
                        ui.horizontal(|ui| {
                            ui.columns_const(|[col_0, col_1, col_2]| {
                                #[cfg(not(target_arch = "wasm32"))]
                                if radio(
                                    col_0,
                                    matches!(self.schema, SchemaLocation::Local(_)),
                                    "Local",
                                ) {
                                    self.schema = SchemaLocation::Local(
                                        std::env::current_dir()
                                            .ok()
                                            .and_then(|p| Some(p.to_str()?.to_string()))
                                            .unwrap_or("/".to_owned()),
                                    );
                                }
                                #[cfg(target_arch = "wasm32")]
                                if radio(
                                    col_0,
                                    matches!(self.schema, SchemaLocation::Worker(_)),
                                    "Local",
                                ) {
                                    self.schema =
                                        SchemaLocation::Worker("Select folder".to_string());
                                }
                                if radio(
                                    col_1,
                                    matches!(self.schema, SchemaLocation::Github(_)),
                                    "GitHub",
                                ) {
                                    self.schema = SchemaLocation::Github(GithubSchemaLocation {
                                        owner: super::DEFAULT_GITHUB_REPO.0.to_string(),
                                        repo: super::DEFAULT_GITHUB_REPO.1.to_string(),
                                        branch: GithubSchemaBranch::Latest,
                                    });
                                }
                                if radio(
                                    col_2,
                                    matches!(self.schema, SchemaLocation::Web(_)),
                                    "Web",
                                ) {
                                    self.schema =
                                        SchemaLocation::Web(super::DEFAULT_SCHEMA_URL.to_string());
                                }
                            });
                        });

                        match &mut self.schema {
                            #[cfg(not(target_arch = "wasm32"))]
                            SchemaLocation::Local(path) => {
                                ui.horizontal(|ui| {
                                    ui.label("Path:");
                                    ui.with_layout(Layout::right_to_left(egui::Align::Min), |ui| {
                                        if ui.button("Browse").clicked()
                                            && let Some(picked_path) = rfd::FileDialog::new()
                                                .pick_folder()
                                                .and_then(|d| d.to_str().map(|s| s.to_owned()))
                                        {
                                            *path = picked_path;
                                        }

                                        ui.add(
                                            egui::TextEdit::singleline(path)
                                                .desired_width(ui.available_width()),
                                        );
                                    });
                                });
                            }

                            #[cfg(target_arch = "wasm32")]
                            SchemaLocation::Worker(name) => {
                                use crate::schema::worker::WorkerProvider;
                                use web_sys::FileSystemPermissionMode;

                                if !*IS_DIRECTORY_PICKER_SUPPORTED {
                                    draw_unsupported_directory_picker(ui);
                                } else {
                                    ui.horizontal(|ui| {
                                        ui.label("Name:");
                                        ui.with_layout(
                                            Layout::right_to_left(egui::Align::Min),
                                            |ui| {
                                                if ui.button("Browse").clicked() {
                                                    self.schema_promises.open_folder_picker(
                                                        FileSystemPermissionMode::Readwrite,
                                                        WorkerProvider::add_folder,
                                                    );
                                                }
                                                egui::ComboBox::from_id_salt("schema_folder")
                                                    .selected_text(name.as_str())
                                                    .width(ui.available_width())
                                                    .show_ui(ui, |ui| {
                                                        match self.schema_promises.get_folder_list(
                                                            WorkerProvider::folders,
                                                        ) {
                                                            None => {
                                                                ui.label("Retrieving...");
                                                            }
                                                            Some(Err(e)) => {
                                                                ui.label(format!(
                                                                    "An error occured: {e}"
                                                                ));
                                                            }
                                                            Some(Ok(entries)) => {
                                                                if entries.is_empty() {
                                                                    ui.label("None");
                                                                } else {
                                                                    for entry in entries {
                                                                        ui.selectable_value(
                                                                            name,
                                                                            entry.0.name(),
                                                                            entry.0.name(),
                                                                        );
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    });
                                            },
                                        );
                                    });
                                }
                            }

                            SchemaLocation::Github(GithubSchemaLocation {
                                owner,
                                repo,
                                branch,
                            }) => {
                                ui.horizontal(|ui| {
                                    ui.columns_const(|[col_owner, col_repo]| {
                                        col_owner.horizontal(|ui| {
                                            ui.label("Owner:");
                                            ui.add(
                                                TextEdit::singleline(owner)
                                                    .desired_width(ui.available_width()),
                                            );
                                        });
                                        col_repo.horizontal(|ui| {
                                            ui.label("Repo:");
                                            ui.add(
                                                TextEdit::singleline(repo)
                                                    .desired_width(ui.available_width()),
                                            );
                                        });
                                    });
                                });

                                // Signing in is keyed on too, so a listing that failed against a
                                // used-up rate limit is retried once there is a token to spend.
                                let key = (owner.clone(), repo.clone(), github_token.is_some());
                                if !owner.is_empty()
                                    && !repo.is_empty()
                                    && self
                                        .github_branch_promise
                                        .as_ref()
                                        .is_none_or(|v| v.0 != key)
                                {
                                    let owner = owner.clone();
                                    let repo = repo.clone();
                                    let github = GithubApi::new(&api_url, github_token.clone());
                                    self.github_branch_promise = Some((
                                        key,
                                        ConvertiblePromise::new_promise(
                                            TrackedPromise::spawn_local(async move {
                                                let branches =
                                                    WebProvider::fetch_github_repository(
                                                        &github, &owner, &repo,
                                                    )
                                                    .await?;
                                                let prs = WebProvider::fetch_github_pull_requests(
                                                    &github, &owner, &repo,
                                                )
                                                .await?;
                                                let mut all_branches = branches;
                                                all_branches.extend(prs);
                                                all_branches.sort();
                                                Ok(all_branches)
                                            }),
                                        ),
                                    ));
                                }

                                ui.horizontal(|ui| {
                                    ui.label("Version:");

                                    if let Some((_, promise)) = &mut self.github_branch_promise {
                                        if let Some(branches) = promise.get_mut(|r| match r {
                                            Ok(vers) => {
                                                self.display_error = None;
                                                Some(vers)
                                            }
                                            Err(e) => {
                                                log::error!("Error fetching versions: {e}");
                                                self.display_error = Some(e);
                                                None
                                            }
                                        }) {
                                            if let Some(branches) = branches {
                                                egui::ComboBox::from_id_salt(
                                                    "setup_github_version",
                                                )
                                                .selected_text(branch.to_string())
                                                .width(ui.available_width())
                                                .show_ui(ui, |ui| {
                                                    let mut branches_latest = vec![];
                                                    let mut branches_version = vec![];
                                                    let mut branches_other = vec![];
                                                    let mut branches_pr = vec![];
                                                    for entry in branches.iter() {
                                                        let vec = match entry {
                                                            GithubSchemaBranch::Latest => {
                                                                &mut branches_latest
                                                            }
                                                            GithubSchemaBranch::Version(_) => {
                                                                &mut branches_version
                                                            }
                                                            GithubSchemaBranch::Other(_) => {
                                                                &mut branches_other
                                                            }
                                                            GithubSchemaBranch::PullRequest {
                                                                ..
                                                            } => &mut branches_pr,
                                                        };
                                                        vec.push(entry);
                                                    }

                                                    if !branches_latest.is_empty() {
                                                        for entry in branches_latest {
                                                            ui.selectable_value(
                                                                branch,
                                                                entry.clone(),
                                                                entry.to_string(),
                                                            );
                                                        }
                                                        ui.separator();
                                                    }

                                                    if !branches_pr.is_empty() {
                                                        for entry in branches_pr {
                                                            ui.selectable_value(
                                                                branch,
                                                                entry.clone(),
                                                                entry.to_string(),
                                                            );
                                                        }
                                                        ui.separator();
                                                    }

                                                    if !branches_other.is_empty() {
                                                        for entry in branches_other {
                                                            ui.selectable_value(
                                                                branch,
                                                                entry.clone(),
                                                                entry.to_string(),
                                                            );
                                                        }
                                                        ui.separator();
                                                    }

                                                    if !branches_version.is_empty() {
                                                        for entry in branches_version {
                                                            ui.selectable_value(
                                                                branch,
                                                                entry.clone(),
                                                                entry.to_string(),
                                                            );
                                                        }
                                                    }
                                                });
                                            } else {
                                                ui.label("Failed to load versions");
                                            }
                                        } else {
                                            ui.label("Loading versions...");
                                        }
                                    } else {
                                        ui.label("No versions available");
                                    }
                                });
                            }

                            SchemaLocation::Web(url) => {
                                ui.horizontal(|ui| {
                                    ui.label("URL:");
                                    ui.add(
                                        TextEdit::singleline(url)
                                            .desired_width(ui.available_width()),
                                    );
                                });
                            }
                        }
                    });

                    ui.add_enabled_ui(self.can_go(), |ui| {
                        ui.add_sized(
                            Vec2::new(ui.available_size_before_wrap().x, 0.0),
                            egui::Button::new("Go"),
                        )
                        .clicked()
                    })
                    .inner
                })
                .inner;

            if is_go_clicked || self.is_startup {
                self.is_startup = false;
                if self.setup_promise.is_none() {
                    let api_url = self.api_url.clone();
                    let location = self.location.clone();
                    let schema = self.schema.clone();
                    self.setup_promise = Some(UnsendPromise::new(async move {
                        let config = BackendConfig {
                            api_url,
                            location,
                            schema,
                        };
                        Backend::new(config.clone())
                            .await
                            .map(|backend| (backend, config))
                    }));
                }
            }
            None
        };

        Modal::default_area("setup-modal".into())
            .order(egui::Order::Middle)
            .show(ctx, |ui| {
                ui.scope_builder(UiBuilder::new().sense(Sense::CLICK | Sense::DRAG), |ui| {
                    egui::containers::Frame::window(ui.style())
                        .show(ui, show_inner)
                        .inner
                })
                .inner
            })
            .inner
    }

    fn can_go(&self) -> bool {
        #[cfg(target_arch = "wasm32")]
        if !*IS_DIRECTORY_PICKER_SUPPORTED
            && (matches!(self.location, InstallLocation::Worker(_))
                || matches!(self.schema, SchemaLocation::Worker(_)))
        {
            return false;
        }

        if matches!(self.location, InstallLocation::Web(_, _))
            && self
                .web_version_promise
                .as_ref()
                .is_none_or(|f| f.1.try_get().map_or(true, |v| v.is_none()))
        {
            return false;
        }
        if matches!(self.schema, SchemaLocation::Github(_))
            && self
                .github_branch_promise
                .as_ref()
                .is_none_or(|f| f.1.try_get().map_or(true, |v| v.is_none()))
        {
            return false;
        }

        true
    }
}

fn radio(ui: &mut egui::Ui, selected: bool, text: impl Into<WidgetText>) -> bool {
    let mut resp = ui
        .vertical_centered_justified(|ui| ui.radio(selected, text))
        .inner;
    if resp.clicked() && !selected {
        resp.mark_changed();
        true
    } else {
        false
    }
}

#[cfg(target_arch = "wasm32")]
type SelectedPickerPromise = UnsendPromise<anyhow::Result<WorkerDirectory>>;

#[cfg(target_arch = "wasm32")]
type FolderListPromise = UnsendPromise<anyhow::Result<Vec<WorkerDirectory>>>;
#[cfg(target_arch = "wasm32")]
type ConvertibleFolderListPromise =
    ConvertiblePromise<FolderListPromise, anyhow::Result<Vec<WorkerDirectory>>>;

#[cfg(target_arch = "wasm32")]
#[derive(Default)]
struct SetupPromises {
    selected: Option<SelectedPickerPromise>,
    list: Option<ConvertibleFolderListPromise>,
}

#[cfg(target_arch = "wasm32")]
static IS_DIRECTORY_PICKER_SUPPORTED: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(SetupPromises::is_supported);

#[cfg(target_arch = "wasm32")]
fn draw_unsupported_directory_picker(ui: &mut egui::Ui) {
    static TITLE: &str = "Your browser does not support the File System Access API.";
    static LINK_DESC: &str = "At the moment, only Chromium-based browsers support it.";
    static LINK: &str = "https://developer.mozilla.org/en-US/docs/Web/API/File_System_Access_API#browser_compatibility";

    ui.vertical_centered(|ui| {
        ui.label(TITLE);
        ui.add(
            egui::Hyperlink::from_label_and_url(
                egui::RichText::new(LINK_DESC).small().weak(),
                LINK,
            )
            .open_in_new_tab(true),
        );
    });
}

#[cfg(target_arch = "wasm32")]
impl SetupPromises {
    fn take_folder(&mut self) -> Option<WorkerDirectory> {
        if let Some(result) = self.selected.take_if(|p| p.ready()) {
            let result = result.block_and_take();

            self.list.take();
            match result {
                Ok(handle) => Some(handle),
                Err(e) => {
                    log::error!("Error picking folder: {e}");
                    None
                }
            }
        } else {
            None
        }
    }

    fn open_folder_picker<F: Future<Output = anyhow::Result<()>>>(
        &mut self,
        mode: web_sys::FileSystemPermissionMode,
        store_folder: impl Fn(WorkerDirectory) -> F + 'static,
    ) {
        use eframe::wasm_bindgen::JsCast;
        use wasm_bindgen_futures::JsFuture;
        use web_sys::{DirectoryPickerOptions, FileSystemDirectoryHandle};

        let ret = UnsendPromise::new(async move {
            let opts = DirectoryPickerOptions::new();
            opts.set_mode(mode);
            let promise = web_sys::window()
                .expect("no window")
                .show_directory_picker_with_options(&opts);
            let promise = promise.map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let result = JsFuture::from(promise).await;
            match result {
                Ok(handle) => {
                    let handle = handle
                        .dyn_into::<FileSystemDirectoryHandle>()
                        .map_err(|_| {
                            anyhow::anyhow!("Error casting to FileSystemDirectoryHandle")
                        })?;
                    let handle = WorkerDirectory(handle);
                    store_folder(handle.clone()).await.map(|()| handle)
                }
                Err(e) => Err(anyhow::anyhow!("Error picking folder: {e:?}")),
            }
        });
        self.selected = Some(ret);
    }

    fn get_folder_list<F: Future<Output = anyhow::Result<Vec<WorkerDirectory>>> + 'static>(
        &mut self,
        future: impl FnOnce() -> F,
    ) -> Option<&anyhow::Result<Vec<WorkerDirectory>>> {
        if self.list.is_none() {
            self.list = Some(ConvertiblePromise::new_promise(
                UnsendPromise::new(future()),
            ));
        }
        self.list.as_mut().unwrap().get(|r| r)
    }

    fn is_supported() -> bool {
        use web_sys::js_sys::Reflect;

        Reflect::has(
            &web_sys::window().expect("no window"),
            &"showDirectoryPicker".into(),
        )
        .expect("Reflect::has failed")
    }
}

/// One row of the version picker: the version on the left, and the patch it belongs to on the
/// right in weak text where the index names one.
fn version_row(
    ui: &mut egui::Ui,
    current: &mut Option<crate::utils::GameVersion>,
    value: Option<crate::utils::GameVersion>,
    text: &str,
    name: Option<&String>,
) {
    let Some(name) = name else {
        ui.selectable_value(current, value, text);
        return;
    };

    let selected = *current == value;
    let response = ui
        .horizontal(|ui| {
            let response = ui.selectable_label(selected, text);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(egui::RichText::new(name).weak());
            });
            response
        })
        .inner;
    if response.clicked() {
        *current = value;
    }
}
