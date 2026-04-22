import { useNavigate } from 'react-router-dom';

const examples = [
  {
    title: "DR Measurement",
    description: "Evaluate image quality and attributes. Upload an image to analyze key metrics including bad crop, padding issues, blurriness, watermark presence, and automatic categorization.",
    link: "/picasso/dr-measurement"
  },
  {
    title: "ROI Detection",
    description: "Identify and extract Regions of Interest (ROI) with precision. Compare results between BASNet (V1) and DETR (V2) models by fine-tuning detection thresholds in real-time.",
    link: "/picasso/roi-detection"
  }
];

const Examples = () => {
  const navigate = useNavigate();

  return (
    <div className="max-w-6xl space-y-8 pb-8">
      <div>
        <h2 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 tracking-tight">Spaces</h2>
        <p className="text-zinc-500 dark:text-zinc-400 mt-2 font-light">Explore demos powered by our customized models.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
        {examples.map((example, index) => (
          <div 
            key={index} 
            onClick={() => example.link && navigate(example.link)}
            className="group bg-white dark:bg-zinc-900/40 border border-black/[0.04] dark:border-white/[0.05] rounded-3xl p-7 shadow-[0_2px_20px_rgba(0,0,0,0.02)] hover:shadow-[0_8px_30px_rgba(0,0,0,0.06)] hover:-translate-y-1 transition-all duration-300 cursor-pointer flex flex-col h-full"
          >
            <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100 mb-3 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors tracking-tight">
              {example.title}
            </h3>
            <p className="text-sm text-zinc-500 dark:text-zinc-400 leading-relaxed flex-1 font-light">
              {example.description}
            </p>
            <div className="mt-8 pt-5 border-t border-black/[0.03] dark:border-white/[0.05] flex items-center justify-between">
              <span className="text-xs font-semibold text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">
                {example.link ? 'View Demo' : 'Coming Soon'}
              </span>
              <span className="text-zinc-300 dark:text-zinc-600 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 group-hover:translate-x-1 transition-all">→</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Examples;