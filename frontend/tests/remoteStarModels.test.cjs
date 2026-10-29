const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const ts = require('typescript')
const source = fs.readFileSync(path.join(__dirname, '../src/remoteStarModels.ts'), 'utf8')
const context = { exports: {} }
vm.runInNewContext(ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText, context)
const { selectHistoricalStarModel, matchesStarModelHistory } = context.exports
const model = { path: '/models/工位一.star', labels: 'OK,NG', conf: 0.7 }
const plain = value => JSON.parse(JSON.stringify(value))

test('history fills any wholly empty card and leaves other models unchanged', () => {
  const existing = { path: '/other.star', labels: 'other', conf: 0.4 }
  const before = [existing, { path: '', labels: '', conf: 0.5 }]
  const result = selectHistoricalStarModel(before, model)
  assert.equal(result.action, 'replaced-empty')
  assert.deepEqual(plain(result.models), [existing, model])
  assert.equal(before[1].path, '')
})

test('history appends without overwriting partially filled model cards', () => {
  const before = [{ path: '', labels: 'unfinished', conf: 0.5 }]
  const result = selectHistoricalStarModel(before, model)
  assert.equal(result.action, 'added')
  assert.deepEqual(plain(result.models), [...before, model])
})

test('same file and labels are not duplicated or silently reconfigured', () => {
  const before = [{ ...model, labels: ' OK , NG ', conf: 0.2 }]
  const result = selectHistoricalStarModel(before, model)
  assert.equal(result.action, 'selected')
  assert.equal(result.models, before)
  assert.equal(result.models[0].conf, 0.2)
  assert.equal(selectHistoricalStarModel(before, { ...model, labels: 'NG,OK' }).action, 'added')
})

test('8-model limit is enforced but existing empty slots may be reused', () => {
  const before = Array.from({ length: 8 }, (_, i) => ({ ...model, path: `/model_${i}.star` }))
  assert.equal(selectHistoricalStarModel(before, model).action, 'full')
  before[4] = { path: '', labels: '', conf: 0.5 }
  assert.equal(selectHistoricalStarModel(before, model).action, 'replaced-empty')
})

test('history search matches filename, complete path, class names and Chinese', () => {
  for (const query of ['', '工位一', '/models', ' ng ', '.STAR']) assert.equal(matchesStarModelHistory(model, query), true)
  assert.equal(matchesStarModelHistory(model, 'missing'), false)
})
