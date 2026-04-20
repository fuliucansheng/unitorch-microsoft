import { useNavigate } from 'react-router-dom';

const examples = [
  {
    title: "DR Measurement",
    description: "This is a demo for DR Measurement. You can upload a fundus image and the output will be the measurements of the optic disc and optic cup.",
    link: "/picasso/dr-measurement"
  },
  {
    title: "ROI Detection",
    description: "This is a demo for detecting regions of interest (ROI) in images using the BASNet model. You can input an image and a mask threshold, and the model will generate an output image with detected ROIs highlighted."
  },
  {
    title: "ROI Detection V2",
    description: "This is a demo for detecting regions of interest (ROI) in images using the DETR model. You can input an image and a score threshold, and the model will generate an output image with detected ROIs highlighted."
  }
];

const Examples = () => {
  const navigate = useNavigate();

  return (
    <div className="max-w-6xl space-y-8 pb-8">
      <div>
        <h2 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 tracking-tight">Examples</h2>
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