#ifndef GEOMETRY_CONSTRUCTOR_HH
#define GEOMETRY_CONSTRUCTOR_HH

#include "G4VUserDetectorConstruction.hh"
#include "MaterialBuilder.hh"
#include "EventProcessor.hh"
#include "ParticleGenerator.hh"
#include "LumaCamMessenger.hh"
#include "G4LogicalVolume.hh"
#include "SimConfig.hh"

class G4VSolid;

class GeometryConstructor : public G4VUserDetectorConstruction {
public:
    GeometryConstructor(ParticleGenerator* gen);
    virtual ~GeometryConstructor();

    virtual G4VPhysicalVolume* Construct();
    void UpdateLShapeGeometry();
    void UpdateScintillatorGeometry(G4double thickness);
    void UpdateSampleGeometry(G4double thickness, G4Material* material, G4double width = Sim::SAMPLE_WIDTH);

private:
    G4double GetArmHalfThickness() const;
    G4double GetScintBackZ() const;
    G4double GetScintCenterZ(G4double thickness) const;
    G4double GetSampleCenterZ() const;
    G4double GetMonitorZ(G4double thickness) const;
    G4VSolid* CreateLShapeSolid() const;

    G4VPhysicalVolume* createWorld();
    G4LogicalVolume* buildLShape(G4LogicalVolume* worldLog);
    void addComponents(G4LogicalVolume* parentLog);

    MaterialBuilder* matBuilder;
    EventProcessor* eventProc;
    G4LogicalVolume* sampleLog;
    G4LogicalVolume* scintLog;
    G4LogicalVolume* blackSideLog; // Added for coating side boxes
    G4LogicalVolume* blackBackLog; // Added for coating back box
    LumaCamMessenger* lumaCamMessenger;
    G4double mirrorCenterZ;
};

#endif
