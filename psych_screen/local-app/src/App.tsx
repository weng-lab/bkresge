import React from 'react';
import { Vitessce } from 'vitessce';
import { myViewConfig } from './ben-config';


console.log('myViewConfig:', myViewConfig);
export default function App() {
  return (
      <Vitessce
        config={myViewConfig}
        theme="dark"
      />
  );
}