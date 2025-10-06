#ifndef GEOMETRY_CONSTRUCTOR_HH
#define GEOMETRY_CONSTRUCTOR_HH

#include "G4VUserDetectorConstruction.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "MaterialBuilder.hh"
#include "LumaCamMessenger.hh"
#include "ParticleGenerator.hh"
#include "EventProcessor.hh"

class GeometryConstructor : public G4VUserDetectorConstruction {
public:
    GeometryConstructor(ParticleGenerator* gen);
    ~GeometryConstructor();

    G4VPhysicalVolume* Construct() override;
    void UpdateSampleGeometry(G4double thickness, G4Material* material);
    void UpdateScintillatorGeometry(G4double thickness);

private:
    G4VPhysicalVolume* createWorld();
    G4LogicalVolume* buildLShape(G4LogicalVolume* worldLog);
    void addComponents(G4LogicalVolume* lShapeLog);

    MaterialBuilder* matBuilder;
    EventProcessor* eventProc;
    G4LogicalVolume* sampleLog;
    G4LogicalVolume* scintLog;
    LumaCamMessenger* lumaCamMessenger;
};

#endif