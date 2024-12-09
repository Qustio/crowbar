use bevy::{math::vec3, prelude::*, winit::{WakeUp, WinitPlugin}};
use camera::PanOrbitState;

mod camera;
mod gltf;
mod vulkano_plugin;

use crate::vulkano_plugin::*;

fn main() {
    App::new()
        // Bevy itself:
        //.add_plugins((WinitPlugin::<WakeUp>::default()))
        .add_plugins(VulkanoPlugin)
        //.add_systems(Startup, camera::setup)
        //.add_systems(Startup, gltf::spawn_gltf)
        // .add_systems(Update,
        //     camera::pan_orbit_camera
        //         .run_if(any_with_component::<PanOrbitState>),
        // )
        .add_systems(Update, create_new_window_system)
        .run();
}

fn create_new_window_system(
    mut commands: Commands,
    r: Option<Res<VulkanApp>>
) {
    // commands.spawn(Window {
    //     resolution: (512.0, 512.0).into(),
    //     present_mode: bevy::window::PresentMode::Fifo,
    //     title: "Secondary window".to_string(),
    //     ..default()
    // });
    match r {
        Some(_) => dbg!("Some"),
        None => dbg!("None"),
    };
}



