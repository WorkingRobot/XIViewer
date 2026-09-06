//! Scratch tool: every vfx a zone places, with the asset it names and where it stands.

use std::io::Cursor;
use std::sync::Arc;

use ironworks::file::layer::{InstanceData, LayerGroup};
use ironworks::file::{File, lgb::LayerGroupFile, sgb::SharedGroupFile};
use ironworks::{
    Ironworks,
    sqpack::{Install, SqPack},
};

const SQPACK: &str = "/home/asriel/.xlcore/ffxiv/game/sqpack";

fn walk<R: ironworks::Resource>(
    ironworks: &Arc<Ironworks<R>>,
    groups: &[LayerGroup],
    depth: u8,
    out: &mut Vec<String>,
) {
    for group in groups {
        for layer in group.layers() {
            for instance in layer.instances() {
                match instance.data() {
                    InstanceData::Vfx(vfx) if !vfx.asset_path().is_empty() => {
                        let at = instance.transform().translation();
                        out.push(format!(
                            "{:<52} auto={} at ({:8.2},{:8.2},{:8.2})",
                            vfx.asset_path(),
                            vfx.auto_play(),
                            at[0],
                            at[1],
                            at[2]
                        ));
                    }
                    InstanceData::SharedGroup(shared)
                        if depth < 6 && !shared.asset_path().is_empty() =>
                    {
                        let Ok(bytes) = ironworks.file::<Vec<u8>>(shared.asset_path()) else {
                            continue;
                        };
                        let Ok(held) = SharedGroupFile::read(Cursor::new(bytes)) else {
                            continue;
                        };
                        walk(ironworks, held.scene().layer_groups(), depth + 1, out);
                    }
                    _ => {}
                }
            }
        }
    }
}

fn main() {
    let sqpack = std::env::var("SQPACK").unwrap_or_else(|_| SQPACK.to_owned());
    let ironworks = Arc::new(Ironworks::new().with_resource(SqPack::new(Install::at_sqpack(sqpack))));
    let level = std::env::args().nth(1).expect("a level directory");
    let mut out = Vec::new();
    for name in ["bg", "planlive", "planmap", "planevent"] {
        let Ok(bytes) = ironworks.file::<Vec<u8>>(&format!("{level}/{name}.lgb")) else {
            continue;
        };
        let Ok(group) = LayerGroupFile::read(Cursor::new(bytes)) else {
            continue;
        };
        walk(&ironworks, std::slice::from_ref(group.group()), 0, &mut out);
    }
    out.sort();
    out.dedup();
    for line in &out {
        println!("{line}");
    }
    eprintln!("{} placed vfx", out.len());
}
