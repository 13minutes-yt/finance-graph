export class IndicatorRegistry {
  constructor(chart, menuContainerId) {
    this.chart = chart;
    this.menuContainer = document.getElementById(menuContainerId);
    this.plugins = new Map();
    this.activeIndicators = new Map(); // pluginId -> { id, pane }
  }

  register(plugin) {
    // plugin: { id, name, pane, calcParams? }
    if (this.plugins.has(plugin.id)) {
      console.warn(`Plugin ${plugin.id} already registered.`);
      return;
    }
    this.plugins.set(plugin.id, plugin);
    this.renderMenuItem(plugin);
  }

  renderMenuItem(plugin) {
    if (!this.menuContainer) {
        console.warn('Menu container not found');
        return;
    }
    const label = document.createElement('label');
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.dataset.indicator = plugin.id;
    
    input.addEventListener('change', (e) => {
      this.toggle(plugin.id, e.target.checked);
    });

    label.appendChild(input);
    label.appendChild(document.createTextNode(` ${plugin.name}`));
    this.menuContainer.appendChild(label);
  }

  toggle(pluginId, enabled) {
    const plugin = this.plugins.get(pluginId);
    if (!plugin) return;

    if (enabled) {
      if (this.activeIndicators.has(pluginId)) return;

      // klinecharts.createIndicator(name, isStack, paneOptions)
      const options = { id: plugin.pane };
      
      // Some indicators might need specific params if provided in plugin definition
      // But for built-ins, we usually just pass the name.
      // If we want to support custom calculated ones later, we might need more logic here.
      
      const id = this.chart.createIndicator(plugin.name, false, options);
      this.activeIndicators.set(pluginId, { id, pane: plugin.pane });
    } else {
      const active = this.activeIndicators.get(pluginId);
      if (!active) return;

      try {
        if (typeof this.chart.removeIndicator === 'function') {
            this.chart.removeIndicator(active.pane, active.id);
        }
      } catch (err) {
         // Fallback for different klinecharts versions or if signature differs
         try {
             this.chart.removeIndicator(active.id);
         } catch (e) {
             console.warn('Failed to remove indicator', e);
         }
      }
      this.activeIndicators.delete(pluginId);
    }
  }
}
