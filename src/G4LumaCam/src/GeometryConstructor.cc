#include "GeometryConstructor.hh"
#include "EventProcessor.hh"
#include "SimConfig.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4UnionSolid.hh"
#include "G4PVPlacement.hh"
#include "G4VisAttributes.hh"
#include "G4OpticalSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4SystemOfUnits.hh"
#include "G4NistManager.hh"
#include "G4SDManager.hh"
#include "G4SubtractionSolid.hh"
#include "LumaCamMessenger.hh"


// Colors for visualization
G4Colour scintColor(0.5, 0.5, 0.5, 0.5);    // Grey color for scintillator
G4Colour blackTapeColor(0.1, 0.1, 0.1);     // Black color for tape
G4Colour mirrorColor(0.5, 0.5, 0.5, 0.5);   // Grey color for mirror
G4Colour sensorColor(1.0, 0.0, 0.0, 0.5);   // Red color for sensor
G4Colour monitorColor(1.0, 0.0, 0.0, 0.5);  // Red color for monitor

/**
 * @brief Constructor for GeometryConstructor.
 * 
 * Initializes the geometry constructor with a particle generator and sets up
 * the event processor and LumaCam messenger. The event processor is responsible
 * for handling events in the simulation, while the LumaCam messenger manages
 * the configuration of the scintillator sample.
 * 
 * @param gen Pointer to the ParticleGenerator object used for generating particles.
 */
GeometryConstructor::GeometryConstructor(ParticleGenerator* gen) 
    : matBuilder(new MaterialBuilder()), eventProc(nullptr), sampleLog(nullptr) {
    eventProc = new EventProcessor("EventProcessor", gen);
    G4SDManager* sdManager = G4SDManager::GetSDMpointer();
    sdManager->AddNewDetector(eventProc);
    G4String filename = "";
    lumaCamMessenger = new LumaCamMessenger(&filename, sampleLog);
}

/**
 * @brief Destructor for GeometryConstructor.
 * 
 * Cleans up the allocated resources, including the material builder and
 * the LumaCam messenger.
 */
GeometryConstructor::~GeometryConstructor() {
    delete matBuilder;
    delete lumaCamMessenger;
}

/**
 * @brief Constructs the geometry of the simulation.
 * 
 * This method creates the world volume and the L-shaped geometry, and places
 * the scintillator sample within it. It also sets up the optical properties
 * and visual attributes for the components. The method returns a pointer to
 * the physical volume of the world.
 * 
 * @return Pointer to the physical volume of the world.
 */
G4VPhysicalVolume* GeometryConstructor::Construct() {
    
    // Create the world volume
    G4VPhysicalVolume* worldPhys = createWorld();
    G4LogicalVolume* worldLog = worldPhys->GetLogicalVolume();

    // Build the L-shaped geometry of the camera box
    G4LogicalVolume* lShapeLog = buildLShape(worldLog);

    // add the sample to the world volume
    constructAndAddSample(worldLog);

    if (lumaCamMessenger) {
        delete lumaCamMessenger;
    }

    G4String filename = "";
    lumaCamMessenger = new LumaCamMessenger(&filename, sampleLog);

    addComponents(lShapeLog);
    return worldPhys;
}

/**
 * @brief Creates the world volume for the simulation.
 * 
 * This method constructs a cubic world volume with a specified size and
 * material properties. The world volume serves as the outermost container
 * for all other geometrical components in the simulation. It is set to be
 * invisible and wireframe for visualization purposes.
 * 
 * @return Pointer to the physical volume of the world.
 */
G4VPhysicalVolume* GeometryConstructor::createWorld() {
    G4Box* worldSolid = new G4Box("WorldSolid", Sim::WORLD_SIZE, Sim::WORLD_SIZE, Sim::WORLD_SIZE);
    G4LogicalVolume* worldLog = new G4LogicalVolume(worldSolid, matBuilder->getVacuum(), "WorldLog");
    G4VPhysicalVolume* worldPhys = new G4PVPlacement(nullptr, G4ThreeVector(), worldLog, "World", nullptr, false, 0);
    
    G4VisAttributes* visAttr = new G4VisAttributes(G4Colour(1.0, 1.0, 1.0));
    visAttr->SetVisibility(false);
    visAttr->SetForceWireframe(true);
    worldLog->SetVisAttributes(visAttr);
    return worldPhys;
}

/**
 * @brief Constructs the L-shaped geometry and places it in the world volume.
 * 
 * This method creates an L-shaped solid using two box solids and a subtraction
 * solid to trim the shape. It also sets up the optical properties of the surface
 * and assigns visual attributes for rendering. The L-shape is placed in the world
 * volume, and its logical volume is returned.
 * 
 * @param worldLog Pointer to the logical volume of the world where the L-shape will be placed.
 * @return Pointer to the logical volume of the L-shape.
 */
G4LogicalVolume* GeometryConstructor::buildLShape(G4LogicalVolume* worldLog) {
    G4Box* arm1 = new G4Box("Arm1", 10*cm, 10*cm, 30*cm);
    G4Box* arm2 = new G4Box("Arm2", 15*cm, 10*cm, 10*cm);
    G4UnionSolid* lShapeSolid = new G4UnionSolid("LShapeSolid", arm1, arm2, nullptr, G4ThreeVector(25*cm, 0, 20*cm));
    G4Box* cutBox = new G4Box("CutBox", 50*cm, 50*cm, 100*cm);
    G4SubtractionSolid* trimmedLShape = new G4SubtractionSolid("TrimmedLShape", lShapeSolid, cutBox, 
                                                               nullptr, G4ThreeVector(0, 0, -100.5*cm));
    G4LogicalVolume* lShapeLog = new G4LogicalVolume(trimmedLShape, matBuilder->getAir(), "LShapeLog");
    new G4PVPlacement(nullptr, G4ThreeVector(), lShapeLog, "LShapePhys", worldLog, false, 0);

    G4OpticalSurface* blackSurf = new G4OpticalSurface("DarkSurface");
    blackSurf->SetType(dielectric_metal);
    blackSurf->SetFinish(ground);
    blackSurf->SetModel(unified);
    G4MaterialPropertiesTable* blackSurfProp = new G4MaterialPropertiesTable();
    G4double PhotonEnergyPVT[12] = {2.08*eV, 2.38*eV, 2.58*eV, 2.7*eV, 2.76*eV, 2.82*eV, 2.92*eV, 2.95*eV, 3.02*eV, 3.1*eV, 3.26*eV, 3.44*eV};
    G4double reflectivity[12] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    G4double efficiency[12] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    blackSurfProp->AddProperty("REFLECTIVITY", PhotonEnergyPVT, reflectivity, 12);
    blackSurfProp->AddProperty("EFFICIENCY", PhotonEnergyPVT, efficiency, 12);
    blackSurf->SetMaterialPropertiesTable(blackSurfProp);
    new G4LogicalSkinSurface("DarkLShape", lShapeLog, blackSurf);

    G4VisAttributes* visAttr = new G4VisAttributes(G4Colour(0.6, 0.3, 0.1));
    visAttr->SetVisibility(true);
    visAttr->SetForceSolid(false);
    lShapeLog->SetVisAttributes(visAttr);
    return lShapeLog;
}


/**
 * @brief Adds the components of the geometry to the logical volume.
 * 
 * This method constructs and places various components of the simulation geometry
 * into the provided logical volume. The primary component is the scintillator volume,
 * which is responsible for converting neutrons to proton recoils and subsequently
 * converting proton recoils to scintillation light. Additional components such as
 * black tape, mirrors, sensors, and monitors are also added to the geometry.
 * 
 * @param lShapeLog Pointer to the logical volume where the components will be added.
 * 
 * Components:
 * - **Scintillator**: A G4Box volume that serves as the main detector for neutron
 *   interactions. It is made of PVT material and is sensitive to events. Optical
 *   properties such as reflectivity and transmittance are defined for its surface.
 * - **Black Tape**: Surrounds the scintillator to block light leakage. Includes
 *   side and back boxes with optical properties defined as dielectric-metal.
 * - **Mirror**: A reflective surface placed at an angle to direct scintillation
 *   light. It is modeled as a polished dielectric-metal surface.
 * - **Sensor**: A cylindrical volume that detects scintillation light. It is a sensitive
 *   volume that represents the input to the lens system of the detector. 
 * - **Monitor**: A flat volume placed above the scintillator to monitor light
 *   transmission. It has defined optical properties for reflectivity and transmittance.
 * 
 * Optical Properties:
 * - The scintillator and monitor surfaces are modeled as dielectric-dielectric
 *   with polished finishes.
 * - The black tape and mirror surfaces are modeled as dielectric-metal with
 *   ground or polished finishes.
 * 
 * Visualization:
 * - Each component is assigned specific visual attributes such as color, transparency,
 *   and visibility to aid in debugging and visualization of the geometry.
 */
void GeometryConstructor::addComponents(G4LogicalVolume* lShapeLog) {
    
    // Scintillator
    G4Box* scintSolid = new G4Box("ScintSolid", Sim::SCINT_SIZE, Sim::SCINT_SIZE, Sim::SCINT_THICKNESS);
    G4VisAttributes* scintVisAttributes = new G4VisAttributes(G4Colour(0.5, 0.5, 0.5, 0.5));
    scintVisAttributes->SetForceSolid(true);
    scintVisAttributes->SetVisibility(true);
    G4LogicalVolume* scintLog = new G4LogicalVolume(scintSolid, matBuilder->getPVT(), "ScintLog");
    new G4PVPlacement(nullptr, G4ThreeVector(0, 0, Sim::SCINT_THICKNESS), scintLog, "ScintPhys", lShapeLog, false, 0);
    scintLog->SetVisAttributes(scintVisAttributes);
    scintLog->SetSensitiveDetector(eventProc);

    G4OpticalSurface* scintSurf = new G4OpticalSurface("ScintSurface");
    scintSurf->SetType(dielectric_dielectric);
    scintSurf->SetFinish(polished);
    scintSurf->SetModel(unified);
    G4MaterialPropertiesTable* surfProp = new G4MaterialPropertiesTable();
    G4double PhotonEnergyPVT[12] = {2.08*eV, 2.38*eV, 2.58*eV, 2.7*eV, 2.76*eV, 2.82*eV,
                                    2.92*eV, 2.95*eV, 3.02*eV, 3.1*eV, 3.26*eV, 3.44*eV};
    G4double reflectivity[12] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    G4double transmittance[12] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    surfProp->AddProperty("REFLECTIVITY", PhotonEnergyPVT, reflectivity, 12);
    surfProp->AddProperty("TRANSMITTANCE", PhotonEnergyPVT, transmittance, 12);
    scintSurf->SetMaterialPropertiesTable(surfProp);
    new G4LogicalSkinSurface("ScintSkinSurface", scintLog, scintSurf);

    // Black tape side boxes
    G4Box* black_side_box = new G4Box("black_side_box", Sim::SCINT_SIZE, Sim::COATING_THICKNESS, Sim::SCINT_THICKNESS);
    G4LogicalVolume* black_side_log = new G4LogicalVolume(black_side_box, matBuilder->getVacuum(), "black_side_log");
    black_side_log->SetVisAttributes(new G4VisAttributes(G4Colour(0.1, 0.1, 0.1)));

    G4ThreeVector placement_top(0, Sim::SCINT_SIZE + Sim::COATING_THICKNESS, Sim::SCINT_THICKNESS);
    new G4PVPlacement(nullptr, placement_top, black_side_log, "black_side_top", lShapeLog, false, 0);

    G4ThreeVector placement_bottom(0, -(Sim::SCINT_SIZE + Sim::COATING_THICKNESS), Sim::SCINT_THICKNESS);
    new G4PVPlacement(nullptr, placement_bottom, black_side_log, "black_side_bottom", lShapeLog, false, 1);

    G4RotationMatrix* sideRotation = new G4RotationMatrix();
    sideRotation->rotateZ(90.*deg);

    G4ThreeVector placement_left(-Sim::SCINT_SIZE - Sim::COATING_THICKNESS, 0, Sim::SCINT_THICKNESS);
    new G4PVPlacement(sideRotation, placement_left, black_side_log, "black_side_left", lShapeLog, false, 2);

    G4ThreeVector placement_right(Sim::SCINT_SIZE + Sim::COATING_THICKNESS, 0, Sim::SCINT_THICKNESS);
    new G4PVPlacement(sideRotation, placement_right, black_side_log, "black_side_right", lShapeLog, false, 3);

    G4Box* black_back_box = new G4Box("black_back_box", Sim::SCINT_SIZE, Sim::SCINT_SIZE, Sim::COATING_THICKNESS);
    G4LogicalVolume* black_back_log = new G4LogicalVolume(black_back_box, matBuilder->getVacuum(), "black_back_log");
    black_back_log->SetVisAttributes(new G4VisAttributes(G4Colour(0.1, 0.1, 0.1)));
    G4ThreeVector placement_back(0, 0, -Sim::COATING_THICKNESS);
    new G4PVPlacement(nullptr, placement_back, black_back_log, "black_back", lShapeLog, false, 4);

    G4OpticalSurface* blackTapeSurf = new G4OpticalSurface("BlackTapeSurface");
    blackTapeSurf->SetType(dielectric_metal);
    blackTapeSurf->SetFinish(ground);
    blackTapeSurf->SetModel(unified);
    G4MaterialPropertiesTable* blackTapeProp = new G4MaterialPropertiesTable();
    G4double blackReflectivity[12] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    G4double blackEfficiency[12] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    blackTapeProp->AddProperty("REFLECTIVITY", PhotonEnergyPVT, blackReflectivity, 12);
    blackTapeProp->AddProperty("EFFICIENCY", PhotonEnergyPVT, blackEfficiency, 12);
    blackTapeSurf->SetMaterialPropertiesTable(blackTapeProp);

    new G4LogicalSkinSurface("black_side_top_surf", black_side_log, blackTapeSurf);
    new G4LogicalSkinSurface("black_side_bottom_surf", black_side_log, blackTapeSurf);
    new G4LogicalSkinSurface("black_side_left_surf", black_side_log, blackTapeSurf);
    new G4LogicalSkinSurface("black_side_right_surf", black_side_log, blackTapeSurf);
    new G4LogicalSkinSurface("black_back_surf", black_back_log, blackTapeSurf);

    // Mirror
    G4Box* mirrorSolid = new G4Box("MirrorSolid", 95*mm, 65*mm, 0.5*um);
    G4LogicalVolume* mirrorLog = new G4LogicalVolume(mirrorSolid, matBuilder->getQuartz(), "MirrorLog");
    G4VisAttributes* mirrorVisAttributes = new G4VisAttributes(G4Colour(0.5, 0.5, 0.5, 0.5));
    mirrorVisAttributes->SetForceSolid(true);
    mirrorVisAttributes->SetVisibility(true);
    G4RotationMatrix* rot = new G4RotationMatrix();
    rot->rotateY(45*deg);
    new G4PVPlacement(rot, G4ThreeVector(0, 0, 20*cm), mirrorLog, "MirrorPhys", lShapeLog, false, 0);
    G4OpticalSurface* mirrorSurf = new G4OpticalSurface("ReflectiveSurface");
    mirrorSurf->SetType(dielectric_metal);
    mirrorSurf->SetFinish(polished);
    mirrorSurf->SetModel(unified);
    mirrorSurf->SetMaterialPropertiesTable(matBuilder->getQuartz()->GetMaterialPropertiesTable());
    new G4LogicalSkinSurface("MirrorSkin", mirrorLog, mirrorSurf);
    mirrorLog->SetVisAttributes(mirrorVisAttributes);

    // Sensor
    // This is a 
    G4Tubs* sensorSolid = new G4Tubs("SensorSolid", 0, 40*mm, 0.5*um, 0*deg, 360*deg);
    G4LogicalVolume* sensorLog = new G4LogicalVolume(sensorSolid, matBuilder->getAir(), "SensorLog");
    G4VisAttributes* sensorVisAttributes = new G4VisAttributes(G4Colour(1.0, 0.0, 0.0, 0.5));
    sensorVisAttributes->SetForceSolid(true);
    sensorVisAttributes->SetVisibility(true);
    G4RotationMatrix* sensorRot = new G4RotationMatrix(); // Renamed to avoid conflict
    sensorRot->rotateY(90*deg);
    new G4PVPlacement(sensorRot, G4ThreeVector(30*cm, 0, 20*cm), sensorLog, "SensorPhys", lShapeLog, false, 0);
    sensorLog->SetVisAttributes(sensorVisAttributes);
    sensorLog->SetSensitiveDetector(eventProc);

    // Monitor
    G4Box* monitorSolid = new G4Box("MonitorSolid", Sim::SCINT_SIZE, Sim::SCINT_SIZE, 0.5*um);
    G4LogicalVolume* monitorLog = new G4LogicalVolume(monitorSolid, matBuilder->getAir(), "MonitorLog");
    G4VisAttributes* monitorVisAttributes = new G4VisAttributes(G4Colour(1.0, 0.0, 0.0, 0.5));
    monitorVisAttributes->SetForceSolid(true);
    monitorVisAttributes->SetVisibility(true);
    new G4PVPlacement(nullptr, G4ThreeVector(0, 0, Sim::SCINT_THICKNESS*2 + 0.5*um), monitorLog, "MonitorPhys", lShapeLog, false, 0);
    monitorLog->SetVisAttributes(monitorVisAttributes);
    monitorLog->SetSensitiveDetector(eventProc);

    G4OpticalSurface* monitorSurf = new G4OpticalSurface("MonitorSurface");
    monitorSurf->SetType(dielectric_dielectric);
    monitorSurf->SetFinish(polished);
    monitorSurf->SetModel(unified);
    G4MaterialPropertiesTable* monitorSurfProp = new G4MaterialPropertiesTable();
    monitorSurfProp->AddProperty("TRANSMITTANCE", PhotonEnergyPVT, transmittance, 12);
    monitorSurfProp->AddProperty("REFLECTIVITY", PhotonEnergyPVT, reflectivity, 12);
    monitorSurf->SetMaterialPropertiesTable(monitorSurfProp);
    new G4LogicalSkinSurface("MonitorSkinSurface", monitorLog, monitorSurf);
}


/**
 * @brief Places a sample in the world volume.
 * 
 * This method creates a solid for the scintillator sample and places it
 * within the world volume. The sample is initially set to be a vacuum, but
 * it can be changed to a different material later if needed. The method
 * also sets up visualization attributes for the sample.
 * 
 * @param worldLog Pointer to the logical volume of the world where the sample will be placed.
 */
void GeometryConstructor::constructAndAddSample(G4LogicalVolume* worldLog) {
    // Create the sample solid
    G4Box* sampleSolid = new G4Box("SampleSolid", Sim::SCINT_SIZE, Sim::SCINT_SIZE, Sim::SAMPLE_THICKNESS);

    // The sample starts off as a vacuum but can be changed to a material later if needed.
    G4NistManager* nistManager = G4NistManager::Instance();
    sampleLog = new G4LogicalVolume(sampleSolid, nistManager->FindOrBuildMaterial("G4_Galactic"), "SampleLog");

    // Set visualization attributes
    G4VisAttributes* sampleVisAttributes = new G4VisAttributes(G4Colour(0.8, 0.2, 0.2, 0.5));
    sampleVisAttributes->SetForceSolid(true);
    sampleVisAttributes->SetVisibility(true);
    sampleLog->SetVisAttributes(sampleVisAttributes);

    // Place the sample in the world volume
    new G4PVPlacement(nullptr, G4ThreeVector(0, 0, -20*cm), sampleLog, "SamplePhys", worldLog, false, 0);
}