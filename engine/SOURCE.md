# engine/

A copy of the game engine from [BambiTP/tagpro-local](https://github.com/BambiTP/tagpro-local)
`engine/` at commit `6cd5df8` (box2d.js, constants.js, game.js, mapLoader.js), unchanged.
Training runs this exact code headless (see `sim/simroom.js`), so the bot learns the real physics.

To update: copy those four files over from tagpro-local, then run `npm test`.
