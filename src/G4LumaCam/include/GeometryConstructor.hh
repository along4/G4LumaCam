#ifndef GEOMETRY_CONSTRUCTOR_HH
#define GEOMETRY_CONSTRUCTOR_HH

#include "G4VUserDetectorConstruction.hh"
#include "MaterialBuilder.hh"
#include "LumaCamMessenger.hh"
#include "G4LogicalVolume.hh"
#include "SimConfig.hh"
#include "G4GenericMessenger.hh"
#include "G4SDManager.hh"

class EventProcessor;
class ParticleGenerator;
class LumaCamMessenger;

class GeometryConstructor : public G4VUserDetectorConstruction {
public:
    GeometryConstructor(ParticleGenerator* gen = nullptr);
    ~GeometryConstructor() override;
    G4VPhysicalVolume* Construct() override;

private:
    MaterialBuilder* matBuilder;
    EventProcessor* eventProc;
    G4LogicalVolume* sampleLog;
    LumaCamMessenger* lumaCamMessenger;

    G4VPhysicalVolume* createWorld();
    G4LogicalVolume* buildLShape(G4LogicalVolume* worldLog);
    void addComponents(G4LogicalVolume* lShapeLog);
};

#endif