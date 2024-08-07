# Diffusers
🤗 Diffusers هي المكتبة الأساسية لنماذج النشر المسبق المتقدمة لتوليد الصور والصوت وحتى الهياكل ثلاثية الأبعاد للجزيئات. سواء كنت تبحث عن حل بسيط للاستنتاج أو تريد تدريب نموذج النشر الخاص بك، 🤗 Diffusers هي مجموعة أدوات قابلة للتطوير تدعم كلا الأمرين. تم تصميم مكتبتنا مع التركيز على [سهولة الاستخدام على الأداء](conceptual/philosophy#usability-over-performance)، [البساطة على السهولة](conceptual/philosophy#simple-over-easy)، و [القابلية للتخصيص على التجريدات](conceptual/philosophy#tweakable-contributorfriendly-over-abstraction).

تتكون المكتبة من ثلاثة مكونات رئيسية:

- خطوط أنابيب النشر المتقدمة للاستدلال باستخدام بضع سطور من التعليمات البرمجية فقط. هناك العديد من خطوط الأنابيب في 🤗 Diffusers، راجع الجدول في نظرة عامة على [خطوط الأنابيب](api/pipelines/overview) للحصول على قائمة كاملة بخطوط الأنابيب المتاحة والمهام التي تحلها.

- [مخططات الضوضاء](api/schedulers/overview) القابلة للتبديل للموازنة بين سرعة وجودة التوليد.

- النماذج [المدربة مسبقًا](api/models) التي يمكن استخدامها كلبنات بناء، والجمع بينها وبين المخططات، لإنشاء أنظمة النشر الخاصة بك من البداية إلى النهاية.

<div class="mt-10">
<div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-2 md:gap-y-4 md:gap-x-5">
<a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./tutorials/tutorial_overview"
><div class="w-full text-center bg-gradient-to-br from-blue-400 to-blue-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">الدروس</div>
<p class="text-gray-700">تعلم المهارات الأساسية التي تحتاجها لبدء إنشاء المخرجات، وبناء نظام النشر الخاص بك، وتدريب نموذج النشر. نوصي بالبدء من هنا إذا كنت تستخدم 🤗 Diffusers لأول مرة!</p>
</a>
<a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./using-diffusers/loading_overview"
><div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">أدلة كيفية الاستخدام</div>
<p class="text-gray-700">أدلة عملية لمساعدتك في تحميل خطوط الأنابيب والنماذج ومخططات الجدولة. ستتعلم أيضًا كيفية استخدام خطوط الأنابيب لمهام محددة، والتحكم في كيفية إنشاء المخرجات، والتحسين لسرعة الاستدلال، وتقنيات التدريب المختلفة.</p>
</a>
<a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./conceptual/philosophy"
><div class="w-full text-center bg-gradient-to-br from-pink-400 to-pink-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">أدلة مفاهيمية</div>
<p class="text-gray-700">افهم سبب تصميم المكتبة بالطريقة التي تم تصميمها بها، وتعرف أكثر على المبادئ التوجيهية الأخلاقية وتنفيذ السلامة لاستخدام المكتبة.</p>
</a>
<a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./api/models/overview"
><div class="w-full text-center bg-gradient-to-br from-purple-400 to-purple-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">مرجع</div>
<p class="text-gray-700">الأوصاف الفنية لكيفية عمل فئات 🤗 Diffusers والطرق.</p>
</a>
</div>
</div>