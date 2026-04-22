import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import Picasso from './pages/picasso-view/Picasso';
import PicassoOverview from './pages/picasso-view/Overview';
import PicassoExamples from './pages/picasso-view/Examples';
import DRMeasurement from './pages/picasso-view/DRMeasurement';
import ROIDetection from './pages/picasso-view/ROIDetection';
import Models from './pages/Models';
import Demos from './pages/Demos';
import Docs from './pages/Docs';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          
          <Route path="picasso" element={<Picasso />}>
            <Route index element={<Navigate to="overview" replace />} />
            <Route path="overview" element={<PicassoOverview />} />
            <Route path="examples" element={<PicassoExamples />} />
            <Route path="dr-measurement" element={<DRMeasurement />} />
            <Route path="roi-detection" element={<ROIDetection />} />
          </Route>

          <Route path="models" element={<Models />} />
          <Route path="demos" element={<Demos />} />
          <Route path="docs" element={<Docs />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;