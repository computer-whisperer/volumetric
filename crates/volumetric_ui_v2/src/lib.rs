//! Aetna-based v2 UI shell for Volumetric.
//!
//! This crate intentionally starts as a separate app path so the current egui UI
//! remains the usable baseline while the Aetna port grows toward parity.

use aetna_core::prelude::*;
use volumetric::Project;

pub const VIEWPORT_KEY: &str = "viewport";
pub const ADD_MODEL_KEY: &str = "action:add-model";
pub const ADD_OPERATOR_KEY: &str = "action:add-operator";
pub const NEW_PROJECT_KEY: &str = "action:new-project";

const MODEL_ROUTE_PREFIX: &str = "model:";
const OPERATOR_ROUTE_PREFIX: &str = "operator:";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectSummary {
    pub imports: usize,
    pub timeline_steps: usize,
    pub exports: usize,
    pub selected_model: Option<String>,
    pub selected_operator: Option<String>,
    pub selected_export: Option<String>,
}

#[derive(Debug)]
pub struct VolumetricUiV2 {
    project: Project,
    selected_model: Option<&'static str>,
    selected_operator: Option<&'static str>,
    selected_export: Option<String>,
    status: String,
}

impl Default for VolumetricUiV2 {
    fn default() -> Self {
        let mut app = Self {
            project: Project::new(),
            selected_model: volumetric_assets::models().first().map(|asset| asset.name),
            selected_operator: volumetric_assets::operators()
                .first()
                .map(|asset| asset.name),
            selected_export: None,
            status: "idle".to_string(),
        };
        app.add_selected_model();
        app
    }
}

impl VolumetricUiV2 {
    pub fn empty() -> Self {
        Self {
            project: Project::new(),
            selected_model: volumetric_assets::models().first().map(|asset| asset.name),
            selected_operator: volumetric_assets::operators()
                .first()
                .map(|asset| asset.name),
            selected_export: None,
            status: "idle".to_string(),
        }
    }

    pub fn project(&self) -> &Project {
        &self.project
    }

    pub fn summary(&self) -> ProjectSummary {
        ProjectSummary {
            imports: self.project.imports().len(),
            timeline_steps: self.project.timeline().len(),
            exports: self.project.exports().len(),
            selected_model: self.selected_model.map(str::to_string),
            selected_operator: self.selected_operator.map(str::to_string),
            selected_export: self.selected_export.clone(),
        }
    }

    fn select_model(&mut self, name: &str) {
        if let Some(asset) = volumetric_assets::get_model(name) {
            self.selected_model = Some(asset.name);
            self.status = format!("selected {}", asset.display_name);
        }
    }

    fn select_operator(&mut self, name: &str) {
        if let Some(asset) = volumetric_assets::get_operator(name) {
            self.selected_operator = Some(asset.name);
            self.status = format!("selected {}", asset.display_name);
        }
    }

    fn add_selected_model(&mut self) {
        let Some(name) = self.selected_model else {
            self.status = "no bundled model available".to_string();
            return;
        };
        let Some(asset) = volumetric_assets::get_model(name) else {
            self.status = format!("missing bundled model {name}");
            return;
        };

        let id = self.project.insert_model(asset.name, asset.bytes.to_vec());
        self.selected_export = Some(id.clone());
        self.status = format!("imported {} as {id}", asset.display_name);
    }

    fn stage_selected_operator(&mut self) {
        let Some(name) = self.selected_operator else {
            self.status = "no bundled operator available".to_string();
            return;
        };
        let Some(asset) = volumetric_assets::get_operator(name) else {
            self.status = format!("missing bundled operator {name}");
            return;
        };

        self.status = format!("staged {}", asset.display_name);
    }
}

impl App for VolumetricUiV2 {
    fn build(&self, _cx: &BuildCx) -> El {
        shell(self)
    }

    fn on_event(&mut self, event: UiEvent) {
        if event.is_click_or_activate(NEW_PROJECT_KEY) {
            self.project = Project::new();
            self.selected_export = None;
            self.status = "new project".to_string();
            return;
        }

        if event.is_click_or_activate(ADD_MODEL_KEY) {
            self.add_selected_model();
            return;
        }

        if event.is_click_or_activate(ADD_OPERATOR_KEY) {
            self.stage_selected_operator();
            return;
        }

        if !matches!(event.kind, UiEventKind::Click | UiEventKind::Activate) {
            return;
        }

        let Some(route) = event.route() else {
            return;
        };

        if let Some(name) = route.strip_prefix(MODEL_ROUTE_PREFIX) {
            self.select_model(name);
        } else if let Some(name) = route.strip_prefix(OPERATOR_ROUTE_PREFIX) {
            self.select_operator(name);
        }
    }
}

pub fn shell(app: &VolumetricUiV2) -> El {
    row([
        left_sidebar(app),
        viewport_workspace(app),
        right_inspector(app),
    ])
    .fill_size()
    .fill(tokens::BACKGROUND)
}

fn left_sidebar(app: &VolumetricUiV2) -> El {
    column([
        column([
            h2("Volumetric").key("brand-title"),
            text("Aetna UI v2").muted().small(),
        ])
        .gap(tokens::SPACE_1),
        divider(),
        section(
            "Project",
            [
                row([
                    button("New").secondary().key(NEW_PROJECT_KEY),
                    button("Open").secondary(),
                ])
                .gap(tokens::SPACE_2)
                .align(Align::Center),
                button("Save").secondary().width(Size::Fill(1.0)),
            ],
        ),
        scroll([
            section(
                "Bundled Models",
                catalog_items(
                    volumetric_assets::models(),
                    MODEL_ROUTE_PREFIX,
                    app.selected_model,
                ),
            ),
            section(
                "Operators",
                catalog_items(
                    volumetric_assets::operators(),
                    OPERATOR_ROUTE_PREFIX,
                    app.selected_operator,
                ),
            ),
        ])
        .key("catalog-scroll")
        .gap(tokens::SPACE_4),
        text(&app.status)
            .muted()
            .small()
            .ellipsis()
            .width(Size::Fill(1.0)),
    ])
    .width(Size::Fixed(tokens::SIDEBAR_WIDTH))
    .height(Size::Fill(1.0))
    .padding(tokens::SPACE_4)
    .gap(tokens::SPACE_4)
    .fill(tokens::CARD)
    .stroke(tokens::BORDER)
}

fn viewport_workspace(app: &VolumetricUiV2) -> El {
    let summary = app.summary();
    column([
        row([
            column([
                h2("Scene").key("scene-title"),
                text("Viewport host region is keyed for custom rendering.")
                    .muted()
                    .small(),
            ])
            .gap(tokens::SPACE_1),
            spacer(),
            badge("Aetna").secondary(),
            badge("wgpu 29").secondary(),
        ])
        .align(Align::Center),
        viewport_placeholder(),
        row([
            badge(format!("{} imports", summary.imports)).muted(),
            badge(format!("{} steps", summary.timeline_steps)).muted(),
            badge(format!("{} exports", summary.exports)).muted(),
            badge(&app.status).success(),
            spacer(),
            button("Frame").secondary(),
        ])
        .align(Align::Center),
    ])
    .width(Size::Fill(1.0))
    .height(Size::Fill(1.0))
    .padding(tokens::SPACE_4)
    .gap(tokens::SPACE_4)
}

fn viewport_placeholder() -> El {
    stack([
        spacer()
            .fill_size()
            .fill(tokens::MUTED)
            .stroke(tokens::BORDER)
            .radius(tokens::RADIUS_MD),
        column([
            h2("3D Viewport").key("viewport-title"),
            text("Renderer host will query this keyed rect and paint behind Aetna chrome.")
                .muted()
                .small(),
        ])
        .gap(tokens::SPACE_2)
        .align(Align::Center)
        .justify(Justify::Center)
        .fill_size(),
    ])
    .key(VIEWPORT_KEY)
    .fill_size()
    .clip()
}

fn right_inspector(app: &VolumetricUiV2) -> El {
    let selected_model = selected_asset_label(volumetric_assets::get_model, app.selected_model);
    let selected_operator =
        selected_asset_label(volumetric_assets::get_operator, app.selected_operator);
    let selected_export = app.selected_export.as_deref().unwrap_or("none");

    column([
        column([
            h2("Inspector"),
            text("Selected catalog entry").muted().small(),
        ])
        .gap(tokens::SPACE_1),
        divider(),
        titled_card(
            "Render Mode",
            [
                row([text("Mode"), badge("ASN v2").secondary()])
                    .align(Align::Center)
                    .justify(Justify::SpaceBetween),
                detail_row("Selected", selected_export),
                detail_row("Status", &app.status),
            ],
        ),
        titled_card(
            "Selection",
            [
                detail_row("Model", selected_model),
                detail_row("Operator", selected_operator),
                button("Add Model")
                    .primary()
                    .width(Size::Fill(1.0))
                    .key(ADD_MODEL_KEY),
                button("Stage Operator")
                    .secondary()
                    .width(Size::Fill(1.0))
                    .key(ADD_OPERATOR_KEY),
            ],
        ),
        spacer(),
        button("Export STL").secondary().width(Size::Fill(1.0)),
    ])
    .width(Size::Fixed(300.0))
    .height(Size::Fill(1.0))
    .padding(tokens::SPACE_4)
    .gap(tokens::SPACE_4)
    .fill(tokens::CARD)
    .stroke(tokens::BORDER)
}

fn section<I>(title: &str, children: I) -> El
where
    I: IntoIterator<Item = El>,
{
    column([
        text(title).muted().small(),
        column(children).gap(tokens::SPACE_2),
    ])
    .gap(tokens::SPACE_2)
}

fn catalog_items(
    assets: &'static [volumetric_assets::BundledAsset],
    route_prefix: &str,
    selected_name: Option<&str>,
) -> Vec<El> {
    assets
        .iter()
        .map(|asset| {
            sidebar_item(
                asset.display_name,
                selected_name == Some(asset.name),
                format!("{route_prefix}{}", asset.name),
            )
        })
        .collect()
}

fn sidebar_item(label: &str, selected: bool, route: String) -> El {
    let item = row([text(label).small().ellipsis().width(Size::Fill(1.0))])
        .align(Align::Center)
        .height(Size::Fixed(tokens::CONTROL_HEIGHT))
        .px(tokens::SPACE_3)
        .py(tokens::SPACE_2)
        .radius(tokens::RADIUS_SM)
        .key(route)
        .cursor(Cursor::Pointer);

    if selected {
        item.fill(tokens::ACCENT)
    } else {
        item
    }
}

fn detail_row(label: &str, value: &str) -> El {
    row([
        text(label),
        spacer(),
        text(value)
            .muted()
            .ellipsis()
            .text_align(TextAlign::End)
            .width(Size::Fixed(150.0)),
    ])
    .align(Align::Center)
}

fn selected_asset_label(
    lookup: fn(&str) -> Option<&'static volumetric_assets::BundledAsset>,
    selected_name: Option<&str>,
) -> &'static str {
    selected_name
        .and_then(lookup)
        .map(|asset| asset.display_name)
        .unwrap_or("none")
}

pub fn shell_bundle(viewport: Rect) -> aetna_core::bundle::artifact::Bundle {
    let app = VolumetricUiV2::default();
    let mut tree = shell(&app);
    aetna_core::bundle::artifact::render_bundle(&mut tree, viewport, Some(env!("CARGO_PKG_NAME")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_reserves_a_viewport_region() {
        let bundle = shell_bundle(Rect::new(0.0, 0.0, 1280.0, 800.0));
        assert!(bundle.tree_dump.contains(VIEWPORT_KEY));
        assert!(bundle.lint.findings.is_empty(), "{}", bundle.lint.text());
    }

    #[test]
    fn default_app_starts_with_a_model_project() {
        let app = VolumetricUiV2::default();
        let summary = app.summary();
        assert_eq!(summary.imports, 1);
        assert_eq!(summary.exports, 1);
        assert!(summary.selected_export.is_some());
    }

    #[test]
    fn add_model_action_appends_unique_import() {
        let mut app = VolumetricUiV2::default();
        app.on_event(UiEvent::synthetic_click(ADD_MODEL_KEY));

        let summary = app.summary();
        assert_eq!(summary.imports, 2);
        assert_eq!(summary.exports, 2);
        assert_eq!(app.project().exports()[0], "simple_sphere_model");
        assert_eq!(app.project().exports()[1], "simple_sphere_model_2");
    }

    #[test]
    fn catalog_click_changes_selected_model() {
        let mut app = VolumetricUiV2::empty();
        app.on_event(UiEvent::synthetic_click("model:simple_torus_model"));

        assert_eq!(
            app.summary().selected_model.as_deref(),
            Some("simple_torus_model")
        );
    }
}
