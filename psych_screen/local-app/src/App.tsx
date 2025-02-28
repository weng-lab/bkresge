import { useState, useEffect } from 'react';
import { Vitessce } from 'vitessce';
import Menu from "./components/Menu";

export default function App(): JSX.Element {
  const [selectedSample, setSelectedSample] = useState<string>("DLPFC_Br8667_mid_manual_alignment_all");
  const [config, setConfig] = useState<object | null>(null);

  // Fetch the config from the Zervers based on the sample name every time a new sample name is selected
  useEffect(() => {
    if (selectedSample) {
      const fetchConfig = async () => {
        let configPath: string;

        if (selectedSample.startsWith("DLPFC")) {
          // Fetch from spatialDLPFC (2024 paper)
          configPath = `https://users.wenglab.org/kresgeb/psych_encode/spatialDLPFC/configs/${selectedSample}/config.json`;
        } else {
          // Fetch from HumanPilot10X (2021 Paper)
          configPath = `https://users.wenglab.org/kresgeb/psych_encode/HumanPilot10X/configs/${selectedSample}/config.json`;
        }

        const response = await fetch(configPath);
        const data = await response.json();
        setConfig(data);
      };
      fetchConfig();
    }
  }, [selectedSample]);

  return (
    <div className="app">
      <Menu sampleSelection={setSelectedSample} />
      <Vitessce
        config={config}
        theme="dark"
      />
    </div>

  );
}