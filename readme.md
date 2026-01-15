repo da pulire i commenti 


ly at ../aten/src/ATen/native/TensorShape.cpp:3609.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
 ┌───────────────────────────────────────────────────────┐
 │ RIEPILOGO PARAMETRI (HAT, DISC, TOT)                  │
 ├───────────────────────────────────────────────────────┤
 │ Componente                │    Parametri │ Dimensione │
 ├───────────────────────────────────────────────────────┤
 │ Parte HAT (Generator)     │   25,877,985 │     98.72 MB │
 │ Discriminatore            │    4,375,745 │     16.69 MB │
 ╞═══════════════════════════════════════════════════════╡
 │ TOTALE SISTEMA            │   46,912,483 │    178.96 MB │
 └───────────────────────────────────────────────────────┘

 *Il Totale include anche le parti ESRGAN/Upsampling del Generatore non listate sopra.
