import React, { useState, useEffect} from 'react';
import { Vitessce } from 'vitessce';
import { myViewConfig } from './ben-config';
import Menu from "./components/Menu";


export default function App(): JSX.Element {
  const [selectedSample, setSelectedSample] = useState<string>("DLPFC_Br8667_mid_manual_alignment_all");
  const [config, setConfig] = useState<object | null>(null);

  useEffect(() => {
    if (selectedSample) {
      const fetchConfig = async () => {
        const response = await fetch(`https://users.wenglab.org/kresgeb/psych_encode_spatialDLPFC/configs/${selectedSample}/config.json`);
        const data = await response.json();
        setConfig(data);
      };
      fetchConfig();
    }
  }, [selectedSample]);

  return (
    <div className="app">
      <Menu sampleSelection={setSelectedSample}/>
      <Vitessce
        config={config}
        theme="dark"
      />
    </div>

  );
}