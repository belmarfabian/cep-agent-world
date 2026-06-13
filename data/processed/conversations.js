// Conversaciones del "modelo mundo". Cada agente (ai, bi) es un encuestado real
// de la CEP N.95; topic indexa labs[]; turns alterna s=0 (agente a) / s=1 (agente b);
// before/after son las opiniones -1..1 antes y despues de conversar.
// demo:true => dialogos ilustrativos escritos a partir de los perfiles reales.
// Para reemplazarlos por conversaciones generadas en vivo con un modelo de
// lenguaje, corre: python scripts/bake_conversations.py  (ver el script).
const CHATS_META={demo:true,modelo:"ejemplo ilustrativo (no generado en vivo)"};
const CHATS=[
{ai:1095,bi:1097,topic:0,before:[-0.80,1.00],after:[-0.62,0.78],turns:[
 {s:0,t:"Mira, yo creo que el sistema de AFP fracasó. La gente cotiza toda la vida y jubila con pensiones de miseria. Tiene que haber un componente de reparto, de solidaridad."},
 {s:1,t:"Pero es que con el reparto lo que pasa es que se lo gastan los políticos, pos. Mi plata es mía. Yo prefiero que esté en mi cuenta y no que la repartan."},
 {s:0,t:"Entiendo el miedo a que el Estado administre mal, es razonable. Pero hoy igual hay comisiones altas y las AFP ganan pase lo que pase. Un sistema mixto podría darte tu cuenta y además un piso solidario."},
 {s:1,t:"Bueno, si es mixto y no me tocan lo que ya ahorré, capaz que sí. Que haya un piso para los que jubilan con nada lo encuentro justo, eso sí."}]},

{ai:1137,bi:1140,topic:1,before:[-0.80,1.00],after:[-0.55,0.80],turns:[
 {s:0,t:"A mí me da pena cuando hablan puro de mano dura. El cabro que roba muchas veces es porque no tuvo na', ni colegio ni pega. Si no atacai eso, sigue igual."},
 {s:1,t:"Mire, señora, yo tengo 84 años y antes uno caminaba tranquilo. Hoy no. La prevención está muy bien, pero mientras tanto uno necesita que la delincuencia se sienta perseguida."},
 {s:0,t:"En eso le encuentro razón, la gente tiene derecho a vivir tranquila. Yo no digo que no haya consecuencias, digo que con pura cárcel no se arregla si después salen peor."},
 {s:1,t:"De acuerdo, las dos cosas entonces. Castigo para el que hace daño, pero también que el Estado llegue antes, con los niños sobre todo. Ahí coincidimos."}]},

{ai:1116,bi:1118,topic:2,before:[-0.80,1.00],after:[-0.60,0.74],turns:[
 {s:0,t:"Yo trabajo en salud y veo migrantes que aportan caleta, hacen pegas que nadie quiere hacer. Cerrar la frontera del todo me parece injusto y poco realista."},
 {s:1,t:"Puede ser, pero acá en la Araucanía igual se siente que entró mucha gente de repente y sin control. No es contra las personas, es que tiene que haber un orden."},
 {s:0,t:"En el orden estamos de acuerdo, la migración tiene que ser regulada, con papeles y registro. Una cosa es ordenar y otra es restringir por restringir."},
 {s:1,t:"Claro, regular no es lo mismo que cerrar. Si entran con sus papeles y trabajan, no tengo drama. Mi tema es el descontrol, no la gente en sí."}]},

{ai:1197,bi:1198,topic:3,before:[-0.80,1.00],after:[-0.58,0.80],turns:[
 {s:0,t:"Yo evalúo mal al gobierno, pero reconozco que algunos problemas vienen de antes. Igual siento que prometieron mucho y avanzaron poco en lo que importa."},
 {s:1,t:"Yo lo evalúo bien dentro de lo posible. Gobernar Chile hoy es difícil, con el Congreso dividido. Han hecho lo que se puede con lo que hay."},
 {s:0,t:"Es cierto que el Congreso traba todo, en eso tienes razón. Quizás soy injusta al cargarle todo al Ejecutivo cuando el problema es más de fondo."},
 {s:1,t:"Y yo capaz peco de complaciente. Hay cosas que sí se podrían haber hecho mejor, sobre todo en gestión. No todo se explica por el Congreso."}]},

{ai:1036,bi:1039,topic:4,before:[-0.80,1.00],after:[-0.62,0.82],turns:[
 {s:0,t:"Para mí el Estado tiene que meterse a reducir la desigualdad. El mercado solo no la arregla, al contrario, la profundiza. Lo veo todos los días acá en mi comuna."},
 {s:1,t:"Yo creo más en que la gente surja con su esfuerzo. Si el Estado interviene mucho, mata el incentivo a trabajar y emprender. Eso al final nos empobrece a todos."},
 {s:0,t:"El esfuerzo importa, no te lo discuto. Pero no todos parten de la misma raya. Un buen sistema premia el esfuerzo y a la vez da piso a los que parten más abajo."},
 {s:1,t:"En eso te encuentro algo de razón. Igualdad de oportunidades sí, sobre todo en educación y salud. De ahí para arriba, que cada uno haga su camino."}]},

{ai:1033,bi:1031,topic:5,before:[-0.80,1.00],after:[-0.60,0.76],turns:[
 {s:0,t:"La educación gratis para todos debería ser un derecho no más. Yo no pude estudiar por plata, y no quiero que a mis nietos les pase lo mismo."},
 {s:1,t:"Entiendo el punto, pero gratis para todos incluye a los que sí pueden pagar. ¿No sería mejor gratis para quien lo necesita y que los demás aporten?"},
 {s:0,t:"Puede ser, mijo, pero cuando pones requisitos al final muchos quedan afuera por trámites. Lo universal es más simple y nadie se siente menos."},
 {s:1,t:"Tiene sentido lo que dice, lo universal evita el estigma. Quizás gratis la base para todos, y que la educación de elite tenga otro financiamiento. Ahí me convence más."}]},

{ai:1018,bi:1016,topic:6,before:[-0.80,1.00],after:[-0.64,0.80],turns:[
 {s:0,t:"Yo priorizo el medioambiente. Acá en la costa vimos cómo se secó todo, y el crecimiento a cualquier costo nos deja sin agua y sin mar limpio."},
 {s:1,t:"Pero en regiones como la mía la pega viene de proyectos que dan trabajo. Si frenamos todo por el medioambiente, ¿de qué vive la gente?"},
 {s:0,t:"No digo frenar todo, digo que los proyectos cumplan reglas ambientales de verdad. Se puede tener pega y cuidar el entorno, no es uno o lo otro."},
 {s:1,t:"En eso concuerdo, lo que no quiero es que se cierre la fuente de trabajo. Si exigen cuidar el ambiente pero dejan trabajar, me parece lo justo."}]}
];
