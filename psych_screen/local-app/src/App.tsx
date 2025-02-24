import React from 'react';
import { Vitessce } from 'vitessce';
import { myViewConfig } from './ben-config';
import Menu from "./components/Menu";


export default function App(): JSX.Element {
  return (
    <div className="app">
      <Menu />
      <Vitessce
        config={myViewConfig}
        theme="dark"
      />
    </div>

  );
}