#include "MaterialBuilder.hh"
#include "G4Element.hh"
#include "G4Isotope.hh"
#include "G4SystemOfUnits.hh"
#include <vector>

MaterialBuilder::MaterialBuilder() : currentScintType(ScintType::EJ200) {
    G4NistManager* nist = G4NistManager::Instance();
    
    // Elements
    G4Element* H = new G4Element("H", "H", 1., 1.01 * g/mole);
    G4Element* C = new G4Element("C", "C", 6., 12.01 * g/mole);
    G4Element* N = new G4Element("N", "N", 7., 14.01 * g/mole);
    G4Element* O = new G4Element("O", "O", 8., 16.00 * g/mole);
    G4Element* Si = new G4Element("Si", "Si", 14., 28.09 * g/mole);
    G4Element* Li = new G4Element("Li", "Li", 3., 6.941 * g/mole); // For GS20
    
    // Vacuum
    vacuum = new G4Material("Vacuum", 1., 1.01 * g/mole, CLHEP::universe_mean_density,
        kStateGas, 2.73 * kelvin, 3.e-18 * pascal);
    
    // Air
    air = new G4Material("Air", 1.290 * mg/cm3, 2);
    air->AddElement(N, 70 * perCent);
    air->AddElement(O, 30 * perCent);
    
    G4double airEnergy[2] = {1.0 * eV, 20.0 * eV};
    G4double airRIndex[2] = {1.0, 1.0};
    G4double airAbs[2] = {100.0 * m, 100.0 * m};
    setupMaterialProperties(air, airEnergy, airRIndex, airAbs, 2);
    
    // Scintillator (PVT)
    scintMaterialPVT = new G4Material("ScintillatorPVT", 1.023 * g/cm3, 2);
    scintMaterialPVT->AddElement(C, 9);
    scintMaterialPVT->AddElement(H, 10);

    const int nPVT = 61;
    G4double pvtEnergy[nPVT] = {
        3.26*eV, 3.25*eV, 3.23*eV, 3.21*eV, 3.20*eV, 3.18*eV, 3.16*eV, 3.15*eV, 3.13*eV, 3.12*eV,
        3.10*eV, 3.08*eV, 3.07*eV, 3.05*eV, 3.04*eV, 3.02*eV, 3.01*eV, 2.99*eV, 2.98*eV, 2.96*eV,
        2.95*eV, 2.94*eV, 2.92*eV, 2.91*eV, 2.90*eV, 2.88*eV, 2.87*eV, 2.85*eV, 2.84*eV, 2.82*eV,
        2.81*eV, 2.80*eV, 2.79*eV, 2.77*eV, 2.76*eV, 2.75*eV, 2.74*eV, 2.73*eV, 2.72*eV, 2.70*eV,
        2.69*eV, 2.68*eV, 2.67*eV, 2.66*eV, 2.65*eV, 2.64*eV, 2.63*eV, 2.62*eV, 2.61*eV, 2.59*eV,
        2.58*eV, 2.57*eV, 2.56*eV, 2.55*eV, 2.54*eV, 2.52*eV, 2.51*eV, 2.50*eV, 2.49*eV, 2.48*eV,
        2.48*eV
    };
    G4double pvtScint[nPVT] = {
        0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.002,
        0.006, 0.016, 0.044, 0.101, 0.173, 0.251, 0.348, 0.454, 0.596, 0.728,
        0.861, 0.949, 0.991, 0.999, 0.982, 0.956, 0.926, 0.890, 0.850, 0.804,
        0.755, 0.706, 0.658, 0.617, 0.582, 0.549, 0.519, 0.492, 0.468, 0.448,
        0.429, 0.410, 0.389, 0.364, 0.337, 0.306, 0.271, 0.238, 0.212, 0.191,
        0.171, 0.153, 0.137, 0.123, 0.109, 0.098, 0.087, 0.077, 0.068, 0.059,
        0.056
    };
    G4double pvtRIndex[nPVT];
    for (int i = 0; i < nPVT; i++) {
        pvtRIndex[i] = 1.58;
    }
    G4double pvtAbs[nPVT];
    for (int i = 0; i < nPVT; i++) {
        pvtAbs[i] = 380.*cm;
    }
    setupMaterialProperties(scintMaterialPVT, pvtEnergy, pvtRIndex, pvtAbs, nPVT, pvtScint);
    scintMaterialPVT->GetIonisation()->SetBirksConstant(0.126 * mm/MeV);
    
    // Scintillator (GS20)
    buildGS20();
    
    // Set default scintillator
    scintMaterial = scintMaterialPVT;
    
    // Graphite
    sampleMaterial = nist->FindOrBuildMaterial("G4_GRAPHITE");
    
    // Quartz Window
    windowMaterial = new G4Material("Quartz", 2.20 * g/cm3, 2);
    windowMaterial->AddElement(Si, 1);
    windowMaterial->AddElement(O, 2);
    
    G4double quartzEnergy[2] = {1.0 * eV, 20.0 * eV};
    G4double quartzRIndex[2] = {1.59, 1.59};
    G4double quartzAbs[2] = {160.0 * cm, 160.0 * cm};
    setupMaterialProperties(windowMaterial, quartzEnergy, quartzRIndex, quartzAbs, 2);
    
    // Black Material
    absorberMaterial = new G4Material("Absorber", 1.290 * mg/cm3, 2);
    absorberMaterial->AddElement(N, 70 * perCent);
    absorberMaterial->AddElement(O, 30 * perCent);
    
    G4double blackEnergy[2] = {1.0 * eV, 20.0 * eV};
    G4double blackRIndex[2] = {1.58, 1.58};
    G4double blackAbs[2] = {0.0 * mm, 0.0 * mm};
    setupMaterialProperties(absorberMaterial, blackEnergy, blackRIndex, blackAbs, 2);
}

void MaterialBuilder::buildGS20() {
    G4NistManager* nist = G4NistManager::Instance();
    
    // Define GS20 material
    G4String mat_name = "ScintillatorGS20";
    scintMaterialGS20 = new G4Material(mat_name, 2.5 * g/cm3, 5);
    
    // Create enriched Lithium (95% 6Li, 5% 7Li)
    G4Element* enriched_li = new G4Element("Enriched_Lithium", "en_Li", 2);
    G4Isotope* li6 = new G4Isotope("6Li", 3, 6, 6.015 * g/mole);
    G4Isotope* li7 = new G4Isotope("7Li", 3, 7, 7.016 * g/mole);
    enriched_li->AddIsotope(li6, 95 * perCent);
    enriched_li->AddIsotope(li7, 5 * perCent);
    
    // Lithium oxide with enriched lithium
    G4Material* enriched_li2o = new G4Material("Enriched_LITHIUM_OXIDE", 2.01 * g/cm3, 2);
    enriched_li2o->AddElement(enriched_li, 2);
    enriched_li2o->AddElement(nist->FindOrBuildElement("O"), 1);
    
    // Cerium oxide
    G4Material* ce2o3 = new G4Material("CERIUM_III_OXIDE", 6.2 * g/cm3, 2);
    ce2o3->AddElement(nist->FindOrBuildElement("Ce"), 2);
    ce2o3->AddElement(nist->FindOrBuildElement("O"), 3);
    
    // GS20 composition
    scintMaterialGS20->AddMaterial(nist->FindOrBuildMaterial("G4_SILICON_DIOXIDE"), 57 * perCent);
    scintMaterialGS20->AddMaterial(nist->FindOrBuildMaterial("G4_ALUMINUM_OXIDE"), 18 * perCent);
    scintMaterialGS20->AddMaterial(nist->FindOrBuildMaterial("G4_MAGNESIUM_OXIDE"), 4 * perCent);
    scintMaterialGS20->AddMaterial(enriched_li2o, 17 * perCent);
    scintMaterialGS20->AddMaterial(ce2o3, 4 * perCent);
    
    // Optical properties
    G4MaterialPropertiesTable* pt20 = new G4MaterialPropertiesTable();
    pt20->AddConstProperty("SCINTILLATIONYIELD", 1255.23 / MeV);
    pt20->AddConstProperty("RESOLUTIONSCALE", 1.);
    pt20->AddConstProperty("FASTTIMECONSTANT", 57. * ns);
    pt20->AddConstProperty("SLOWTIMECONSTANT", 98. * ns);
    pt20->AddConstProperty("YIELDRATIO", 1.);
    
    G4double ScintEnergy[] = {
        1.771147508*eV, 1.785693715*eV, 1.806886491*eV, 1.835205916*eV, 1.859861711*eV, 1.887526235*eV,
        1.911215667*eV, 1.937982147*eV, 1.968038263*eV, 1.993817314*eV, 2.025656659*eV, 2.052965257*eV,
        2.086731237*eV, 2.11868072*eV, 2.14858044*eV, 2.176213997*eV, 2.214200693*eV, 2.243574408*eV,
        2.270347301*eV, 2.294304242*eV, 2.322309111*eV, 2.354650359*eV, 2.376721902*eV, 2.406788116*eV,
        2.433745705*eV, 2.469279938*eV, 2.493591512*eV, 2.522559239*eV, 2.556474115*eV, 2.586920261*eV,
        2.609157128*eV, 2.627217637*eV, 2.650155755*eV, 2.673476749*eV, 2.692441907*eV, 2.711710765*eV,
        2.726355401*eV, 2.746080968*eV, 2.761100218*eV, 2.77121984*eV, 2.796776839*eV, 2.817562006*eV,
        2.828136098*eV, 2.8547708*eV, 2.871018238*eV, 2.892938281*eV, 2.898532973*eV, 2.915258618*eV,
        2.926567493*eV, 2.937990045*eV, 2.955188453*eV, 2.966796809*eV, 2.978496723*eV, 2.990276024*eV,
        2.996253925*eV, 3.008174378*eV, 3.020203583*eV, 3.038367271*eV, 3.050653453*eV, 3.069200332*eV,
        3.081737652*eV, 3.100651336*eV, 3.139096176*eV, 3.191743043*eV, 3.225507311*eV, 3.239154774*eV,
        3.245982849*eV, 3.259772991*eV, 3.273696694*eV, 3.2806075*eV, 3.294726235*eV, 3.301758535*eV,
        3.308837152*eV, 3.323134953*eV, 3.330272736*eV, 3.337474277*eV, 3.344707031*eV, 3.351971202*eV,
        3.359216802*eV, 3.366443383*eV, 3.366544205*eV, 3.381295329*eV, 3.396090789*eV, 3.396227597*eV,
        3.411206068*eV, 3.418692972*eV, 3.426230217*eV, 3.441598022*eV, 3.449219077*eV, 3.464740871*eV,
        3.480313198*eV, 3.495989894*eV, 3.496098625*eV, 3.520093022*eV, 3.528102993*eV, 3.544381799*eV,
        3.560849122*eV, 3.569083637*eV, 3.60277172*eV, 3.654582691*eV, 3.735258252*eV, 3.829204446*eV,
        3.927975404*eV, 4.010757254*eV, 4.130449603*eV, 4.245659698*eV, 4.367453005*eV, 4.470036937*eV,
        4.577524753*eV, 4.763720755*eV, 4.886025126*eV, 5.06482313*eV, 5.293804937*eV, 5.464820789*eV,
        5.64730209*eV, 5.864795368*eV, 6.149039422*eV
    };
    const G4int nPoints = sizeof(ScintEnergy)/sizeof(G4double);
    G4double ScintFast[nPoints] = {
        0.004514673, 0.006772009, 0.006772009, 0.009029345, 0.006772009, 0.004514673, 
        0.002257336, 0.004514673, 0.002257336, 0.004514673, 0.006772009, 0.004514673, 
        0.004514673, 0.006772009, 0.006772009, 0.004514673, 0.006772009, 0.009029345, 
        0.011286682, 0.013544018, 0.015801354, 0.020316027, 0.0248307, 0.027088036, 
        0.033860045, 0.036117381, 0.047404063, 0.058690745, 0.065462754, 0.074492099, 
        0.090293454, 0.101580135, 0.11738149, 0.128668172, 0.139954853, 0.158013544, 
        0.173814898, 0.18510158, 0.200902935, 0.214446953, 0.23476298, 0.250564334, 
        0.270880361, 0.293453725, 0.311512415, 0.329571106, 0.34537246, 0.358916479, 
        0.376975169, 0.399548533, 0.415349887, 0.431151242, 0.446952596, 0.460496614, 
        0.476297968, 0.489841986, 0.505643341, 0.519187359, 0.53724605, 0.553047404, 
        0.571106095, 0.584650113, 0.598194131, 0.598194131, 0.591422122, 0.58013544, 
        0.568848758, 0.553047404, 0.539503386, 0.519187359, 0.507900677, 0.492099323, 
        0.478555305, 0.458239278, 0.440180587, 0.426636569, 0.413092551, 0.399548533, 
        0.379232506, 0.35214447, 0.365688488, 0.338600451, 0.300225734, 0.318284424, 
        0.286681716, 0.264108352, 0.243792325, 0.227990971, 0.205417607, 0.182844244, 
        0.148984199, 0.110609481, 0.124153499, 0.09255079, 0.074492099, 0.056433409, 
        0.042889391, 0.029345372, 0.018058691, 0.009029345, 0.006772009, 0.006772009, 
        0.004514673, 0.004514673, 0.004514673, 0.006772009, 0.006772009, 0.006772009, 
        0.004514673, 0.004514673, 0.004514673, 0.004514673, 0.006772009, 0.006772009, 
        0.009029345, 0.006772009, 0.006772009
    };
    G4double r_ind[nPoints];
    for (int i = 0; i < nPoints; i++) {
        r_ind[i] = 1.55;
    }
    G4double abs[nPoints];
    for (int i = 0; i < nPoints; i++) {
        abs[i] = 100 * cm;
    }
    pt20->AddProperty("FASTCOMPONENT", ScintEnergy, ScintFast, nPoints);
    pt20->AddProperty("RINDEX", ScintEnergy, r_ind, nPoints);
    pt20->AddProperty("ABSLENGTH", ScintEnergy, abs, nPoints);
    
    scintMaterialGS20->SetMaterialPropertiesTable(pt20);
}

void MaterialBuilder::setScintillatorType(ScintType type) {
    currentScintType = type;
    if (type == ScintType::EJ200) {
        scintMaterial = scintMaterialPVT;
    } else if (type == ScintType::GS20) {
        scintMaterial = scintMaterialGS20;
    }
}

void MaterialBuilder::setScintillatorType(const G4String& typeName) {
    if (typeName == "EJ200") {
        setScintillatorType(ScintType::EJ200);
    } else if (typeName == "GS20") {
        setScintillatorType(ScintType::GS20);
    } else {
        G4cerr << "Unknown scintillator type: " << typeName << ". Available types: EJ200, GS20" << G4endl;
    }
}

void MaterialBuilder::setupMaterialProperties(G4Material* mat, const G4double* energies,
    const G4double* rindex, const G4double* abslength,
    int nEntries, const G4double* scintillation) {
    std::vector<G4double> energiesCopy(energies, energies + nEntries);
    std::vector<G4double> rindexCopy(rindex, rindex + nEntries);
    std::vector<G4double> abslengthCopy(abslength, abslength + nEntries);
    
    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
    
    mpt->AddProperty("RINDEX", energiesCopy.data(), rindexCopy.data(), nEntries);
    mpt->AddProperty("ABSLENGTH", energiesCopy.data(), abslengthCopy.data(), nEntries);
    
    if (scintillation) {
        std::vector<G4double> scintillationCopy(scintillation, scintillation + nEntries);
        mpt->AddProperty("FASTCOMPONENT", energiesCopy.data(), scintillationCopy.data(), nEntries);
        if (mat == scintMaterialPVT) {
            mpt->AddConstProperty("SCINTILLATIONYIELD", 10000./MeV);
            mpt->AddConstProperty("RESOLUTIONSCALE", 1.0);
            mpt->AddConstProperty("SCINTILLATIONTIMECONSTANT", 2.1 * ns);
            mpt->AddConstProperty("SCINTILLATIONRISETIME", 0.9 * ns);
            mpt->AddConstProperty("FASTTIMECONSTANT", 3.2 * ns);
            mpt->AddConstProperty("YIELDRATIO", 0.);
        } else if (mat == scintMaterialGS20) {
            mpt->AddConstProperty("SCINTILLATIONYIELD", 1255.23 / MeV);
            mpt->AddConstProperty("RESOLUTIONSCALE", 1.);
            mpt->AddConstProperty("FASTTIMECONSTANT", 57. * ns);
            mpt->AddConstProperty("SLOWTIMECONSTANT", 98. * ns);
            mpt->AddConstProperty("YIELDRATIO", 1.);
        }
    }
    
    mat->SetMaterialPropertiesTable(mpt);
}